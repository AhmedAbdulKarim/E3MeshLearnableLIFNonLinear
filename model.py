import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate

__all__ = ['QKFormer']



class NonLinearParametricLIF(nn.Module):
    def __init__(self, init_tau=2.0, init_v_threshold=1.0, init_w_res=0.0):
        super().__init__()
        
        # 1. The Learnable Parameters (Addressing the accuracy gap)
        self.w_decay = nn.Parameter(torch.tensor([1.0 / init_tau], dtype=torch.float32))
        self.v_threshold = nn.Parameter(torch.tensor([init_v_threshold], dtype=torch.float32))
        self.w_res = nn.Parameter(torch.tensor([init_w_res], dtype=torch.float32))
        
        self.v = 0.0
        self.surrogate_function = surrogate.Sigmoid(alpha=5.0)

    def reset(self):
        self.v = 0.0

    def forward(self, x):
        T = x.shape[0]
        out = []

        if isinstance(self.v, float) and self.v == 0.0:
            self.v = torch.zeros_like(x[0])

        safe_reset_weight = torch.sigmoid(self.w_res)
        
        # We allow the network to learn both positive and negative decay paths
        # No sigmoid clamp here, letting the network explore complex temporal weights
        learned_decay = self.w_decay 

        for t in range(T):
            # A. The Non-Linear State Transition (The RNN Equivalent)
            # F.silu (Swish: x * sigmoid(x)) acts as a self-gating mechanism.
            # If the residual voltage is useful, it passes through. If it's negative noise, it gets squashed.
            self.v = F.silu(self.v * learned_decay) + x[t]

            # B. Fire (with Learnable Threshold)
            spike = self.surrogate_function(self.v - self.v_threshold)
            out.append(spike)

            # C. Multiplicative Reset (with Detach)
            spike_d = spike.detach()
            self.v = self.v * (1.0 - spike_d) + (self.v * safe_reset_weight) * spike_d

        return torch.stack(out, dim=0)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        #self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp1_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        #self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.mlp2_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)
        return x
    
    
    
class HierarchicalSpikingMesh(nn.Module):
    def __init__(self, dim, sparsity_level='high'):
        super().__init__()
        
        # Determine Connectivity based on hierarchy level
        if sparsity_level == 'high':
            # Stage 1: "Least connected, convolution like, parameter reuse"
            # groups=dim means Depthwise Convolution. Strict local spatial routing only.
            groups = dim
            kernel = 3
        elif sparsity_level == 'medium':
            # Stage 2: Intermediate mixing
            # Tokens share data within sub-groups (e.g., 4 channels per group)
            groups = dim // 4 
            kernel = 3
        else: # 'low'
            # Stage 3/4: Dense/Global routing (The Attention Equivalent)
            # groups=1 means fully dense matrix multiplication across all channels
            groups = 1
            kernel = 1 # 1x1 dense mix to prevent parameter explosion
            
        # The Internal Mesh Layers (3 layers deep with strict skip connections)
        self.mesh1 = nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel//2, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.lif1 = NonLinearParametricLIF(init_tau=2.0)
        
        self.mesh2 = nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel//2, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.lif2 = NonLinearParametricLIF(init_tau=2.0)
        
        self.mesh3 = nn.Conv2d(dim, dim, kernel_size=1, bias=False) # Final aggregation hop
        self.bn3 = nn.BatchNorm2d(dim)
        self.lif3 = NonLinearParametricLIF(init_tau=2.0)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.flatten(0, 1)

        # --- Micro-Layer 1 ---
        out1 = self.mesh1(x_flat)
        out1 = self.bn1(out1).reshape(T, B, C, H, W)
        spikes1 = self.lif1(out1)
        
        # Skip Connection 1 (Input bypasses Micro-Layer 1)
        res1 = x + spikes1

        # --- Micro-Layer 2 ---
        out2 = self.mesh2(res1.flatten(0, 1))
        out2 = self.bn2(out2).reshape(T, B, C, H, W)
        spikes2 = self.lif2(out2)
        
        # Skip Connection 2 (Res1 bypasses Micro-Layer 2)
        res2 = res1 + spikes2
        
        # --- Micro-Layer 3 ---
        out3 = self.mesh3(res2.flatten(0, 1))
        out3 = self.bn3(out3).reshape(T, B, C, H, W)
        spikes3 = self.lif3(out3)
        
        # Skip Connection 3 (Res2 bypasses Micro-Layer 3)
        final_out = res2 + spikes3

        return final_out





class HierarchicalMeshSNN(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, embed_dims=256, num_classes=11):
        super().__init__()
        
        # Standard Patch Embedding (Extracting the initial tokens)
        self.patch_embed = PatchEmbeddingStage(img_size_h, img_size_w, in_channels, embed_dims)
        
        # --- THE HIERARCHY ---
        
        # STAGE 1: High Sparsity (Local Texture & Edge Tracking)
        # Replaces early Token Attention
        self.stage1_mesh = HierarchicalSpikingMesh(embed_dims, sparsity_level='high')
        self.stage1_mlp = MLP(embed_dims, embed_dims * 4) # Keeping standard MLPs for feature expansion
        
        # STAGE 2: Medium Sparsity (Sub-gesture formulation)
        self.stage2_mesh = HierarchicalSpikingMesh(embed_dims, sparsity_level='medium')
        self.stage2_mlp = MLP(embed_dims, embed_dims * 4)
        
        # STAGE 3: Low Sparsity (Dense Global Recognition)
        # Replaces Spiking Self Attention
        self.stage3_mesh = HierarchicalSpikingMesh(embed_dims, sparsity_level='low')
        self.stage3_mlp = MLP(embed_dims, embed_dims * 4)
        
        # Classification Head
        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x):
        # x: [T, B, C, H, W]
        x = self.patch_embed(x)
        
        # Forward through Hierarchy
        x = x + self.stage1_mesh(x)
        x = x + self.stage1_mlp(x)
        
        x = x + self.stage2_mesh(x)
        x = x + self.stage2_mlp(x)
        
        x = x + self.stage3_mesh(x)
        x = x + self.stage3_mlp(x)
        
        # Global Average Pooling and Classify
        x = x.flatten(3).mean(3) # Pool spatial dimensions
        x = self.head(x.mean(0)) # Pool temporal dimension and classify
        return x



class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        #self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.proj_lif=NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF
        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF


    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        # x_feat = x.flatten(0, 1)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//8, W//8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat # shortcut

        return x

class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = NonLinearParametricLIF(tau=2.0, v_threshold=1.0, init_w_res=0.0) # Replacing standard LIF with the novel NonLinearParametricLIF

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

# --- 1. The Block Wrapper ---
# This holds our new Mesh and the standard MLP, replacing the old Transformer blocks.
class SpikingMeshBlock(nn.Module):
    def __init__(self, dim, sparsity_level, mlp_ratio=4.):
        super().__init__()
        self.mesh = HierarchicalSpikingMesh(dim=dim, sparsity_level=sparsity_level)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        # Heavy residual skip connections around the major blocks
        x = x + self.mesh(x)
        x = x + self.mlp(x)
        return x

# --- 2. The Upgraded vit_snn Macro-Architecture ---
class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None, in_chans=3, no_weight_decay=None, **kwargs
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        # --- STAGE 1: High Sparsity (Local Feature Extraction) ---
        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2)

        # Replacing TokenSpikingTransformer with our Sparse Mesh Block
        stage1 = nn.ModuleList([
            SpikingMeshBlock(dim=embed_dims // 2, sparsity_level='high', mlp_ratio=mlp_ratios)
            for _ in range(1) # Keeping your original range logic
        ])

        # --- STAGE 2: Low Sparsity (Global Dense Mixing) ---
        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)

        # Replacing SpikingTransformer with our Dense Mesh Block
        stage2 = nn.ModuleList([
            SpikingMeshBlock(dim=embed_dims, sparsity_level='low', mlp_ratio=mlp_ratios)
            for _ in range(1)
        ])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage2", stage2)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    # --- CRITICAL FIX: Protecting the new Physics Parameters ---
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pose_embed'}
        for name, _ in self.named_parameters():
            # If we don't protect these, AdamW will crush the nonlinear physics back to 0
            if 'w_res' in name or 'w_decay' in name or 'v_threshold' in name:
                nwd.add(name)
        return nwd

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        # Route through Stage 1 (Sparse)
        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        # Route through Stage 2 (Dense)
        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

@register_model
def QKFormer(pretrained=False, **kwargs):
    # Gracefully extract num_classes from kwargs if timm passes it, otherwise default to 11
    dynamic_classes = kwargs.pop('num_classes', 11)
    
    model = vit_snn(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=dynamic_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

#@register_model
#def QKFormer(pretrained=False, **kwargs):
#    model = vit_snn(
#        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
#        in_channels=2, num_classes=11, qkv_bias=False,
#        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
#        **kwargs
#    )
#    model.default_cfg = _cfg()
#    return model


from timm.models import create_model

if __name__ == '__main__':
    x = torch.randn(1, 1, 2, 128, 128).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    ).cuda()
    model.eval()

    from torchinfo import summary
    summary(model, input_size=(1, 1, 2, 128, 128))
    # y = model(x)
    # print(y.shape)
    # print('Test Good!')










# the changes made in this code are:
# 1. Added a new argument `num_classes` to the `QKFormer` function to allow dynamic specification of the number of output classes.
# 2. Updated the `vit_snn` class to accept `num_classes` as an argument and use it to define the classification head accordingly.
# 3. Modified the `QKFormer` function to extract `num_classes` from `kwargs` and pass it to the `vit_snn` class when creating the model instance. This allows for flexibility in specifying the number of
#    classes when creating the model, while still providing a default value of 11 if `num_classes` is not specified in `kwargs`.
# These changes enable the `QKFormer` model to be more adaptable to different classification tasks by allowing users to specify the number of output classes as needed.
# 4. backend='cupy' -> backend='torch' in all MultiStepLIFNode instances to ensure compatibility with PyTorch.
# 5. Added comments to explain the changes made in the code for better clarity and understanding.
# 6. Th train.py file has been updated
# 7. The new command for training the model with Adam's optimizer and a learning rate of 0.001 will be added to the notebook on Kaggle.
# 8. **kwargs was added to the model constructor vit_snn


# Change Log:
# non-linear parametric LIF neuron with learnable decay, threshold, and reset parameters and non-linear state transition function (F.silu) to enhance temporal dynamics and accuracy
# hierarchical spiking mesh with varying sparsity levels (high, medium, low) to replace attention mechanisms
# skip connections at multiple levels of the mesh to improve gradient flow and feature reuse
# enhanced patch embedding with multi-stage convolutional downsampling and residual connections

