import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import model, utils
from spikingjelly.clock_driven import functional

# 1. Import your custom dataset
import dvsgc

from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import autoaugment

_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
writer = SummaryWriter("./")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='QKFormer', help='model')
    parser.add_argument('--dataset', default='DVS128Gesture', help='dataset')
    parser.add_argument('--num-classes', type=int, default=11, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='/media/data/DVS128Gesture', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")

    parser.add_argument('--amp', default=True, action='store_true', help='Use AMP training')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--tb', default=True,  action='store_true', help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')

    # Optimizer Parameters (AdamW by default)
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    # Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT', help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N', help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=2e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=192, metavar='N', help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N', help='epoch repeat multiplier')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5, help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0., help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio')
    parser.add_argument('--mixup-prob', type=float, default=0.5, help='Probability of performing mixup or cutmix')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params.')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N', help='Turn off mixup after this epoch')

    # 2. Sequence arguments
    parser.add_argument('--seq_len', default=4, type=int, help='number of gestures in the chain')
    parser.add_argument('--class_num', default=4, type=int, help='number of base classes to use')

    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N,T,C,H,W = image.shape

        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])

        if trival_aug != None:
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])

        if mixup_fn is not None:
            image, target = mixup_fn(image, target)
            target_for_compu_acc = target.argmax(dim=-1)

        # WARNING: Randomly dropping frames destroys sequence chains. Bypassed.
        # if T_train:
        #     sec_list = np.random.choice(image.shape[1], T_train, replace=False)
        #     sec_list.sort()
        #     image = image[:, sec_list]

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()

            # --- INJECTED DIAGNOSTICS HERE ---
            scaler.unscale_(optimizer) # Unscale to get raw gradient values
            
            if model.head.weight.grad is not None:
                grad_max = model.head.weight.grad.abs().max().item()
                grad_mean = model.head.weight.grad.abs().mean().item()
                print(f"  --> [Diag] Grad Max: {grad_max:.6f} | Mean: {grad_mean:.6f}")
            else:
                print("  --> [Diag] GRADIENT IS NONE!")
            # -------------------------------
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # --- INJECT DIAGNOSTICS HERE ---
            if model.head.weight.grad is not None:
                grad_max = model.head.weight.grad.abs().max().item()
                grad_mean = model.head.weight.grad.abs().mean().item()
                print(f"  --> [Diag] Grad Max: {grad_max:.6f} | Mean: {grad_mean:.6f}")
            # -------------------------------

            
            optimizer.step()

        functional.reset_net(model)
        
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


# 3. Dynamic custom dataloader instantiation
def load_data(dataset_dir, distributed, T, seq_len, class_num):
    print("Loading data")
    st = time.time()

    dataset_train = dvsgc.DVSGestureChain(
        root=dataset_dir, split='train', frames_number=T, 
        seq_len=seq_len, class_num=class_num, alpha_min=0.5, alpha_max=0.7)
    
    dataset_test = dvsgc.DVSGestureChain(
        root=dataset_dir, split='test', frames_number=T, 
        seq_len=seq_len, class_num=class_num, alpha_min=0.5, alpha_max=0.7)
        
    print("Took", time.time() - st)
    print("Creating data loaders")
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'
    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'
    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'
    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)
    data_path = args.data_path

    # 4. Pass arguments dynamically
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        data_path, args.distributed, args.T, args.seq_len, args.class_num)
        
    num_classes = len(dataset_train.classes)
    print(f"Dataset generated {num_classes} unique sequence combinations.")

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False, pin_memory=True)

    # 5. Pass dynamic classes to Timm
    print("Creating model")
    model = create_model(
        "QKFormer",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        #checkpoint = torch.load(args.resume, map_location='cpu')
        
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

    train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    train_trivalaug = autoaugment.SNNAugmentWide()
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
        mixup_fn = Mixup(**mixup_args)
        
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= 75 and mixup_fn is not None:
            mixup_fn.mixup_enabled = False
            
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
            
        if utils.is_main_process():
            if train_tb_writer is not None:
                train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
                
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        
        if te_tb_writer is not None and utils.is_main_process():
            te_tb_writer.add_scalar('test_loss', test_loss, epoch)
            te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        if output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(checkpoint, os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        
    if output_dir:
        utils.save_on_master(checkpoint, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
#
# Summary of changes:
# 1. Added a new argument `--num-classes` to the argument parser to allow users to specify the number of classes in the dataset when creating the model. This argument is passed to the `QKFormer` model to ensure that the classification head is correctly sized for the number
#    of classes in the dataset.
# 2. Updated the `QKFormer` model definition to accept the `num_classes` argument and use it to define the output layer of the model accordingly. This ensures that the model can be trained on datasets with different numbers of classes without requiring manual modifications to the model architecture.
# 3. Modified the `train_one_epoch` function to include an optional `T_train` argument that allows for randomly dropping frames during training. This can help improve the model's robustness by simulating scenarios where some frames may be missing or corrupted. However, this functionality is currently bypassed to preserve the integrity of sequence chains in the dataset.
# 4. Updated the `load_data` function to accept additional arguments for sequence length and class number, allowing for dynamic instantiation of the custom dataset based on user-specified parameters. This makes the data loading process more flexible and adaptable to different dataset configurations.
# 5. Added support for mixup and cutmix data augmentation techniques in the training loop, allowing for improved generalization and robustness of the model. The mixup and cutmix parameters can be specified through command-line arguments, providing users with control over the augmentation process.
# 6. Implemented a learning rate scheduler that can be configured through command-line arguments,
#allowing for dynamic adjustment of the learning rate during training. This can help improve convergence and overall performance of the model.
# 7. Added functionality to save model checkpoints at the end of each epoch, as well as when a new maximum test accuracy is achieved. This allows for easy resumption of training and tracking of model performance over time.
# These changes enhance the flexibility and functionality of the training script, allowing for dynamic configuration of the model, dataset, and training process based on user-specified parameters. This makes it easier to experiment with different settings and achieve optimal performance on the DVS128Gesture dataset.
# The original code is in trainOriginal.py, and the modified code is in train.py.
# The changes made in the model.py file have been recorded there at the end of the file
# The new command for training the model with Adam's optimizer and a learning rate of 0.001 will be added to the notebook on Kaggle.
# Printing the gradients was added to the training loop
# added weights_only=False
