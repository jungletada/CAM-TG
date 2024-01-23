import os
import argparse
import datetime
import time
import json
import random
import torch
import logging
import numpy as np
import torch.nn as nn
from misc import pyutils, torchutils
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

import voc12.dataloader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import utils
from utils import logger_info, data_mkdir
from engine import evaluate
from net.mctg_cam import MCTG

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-per-gpu', default=16, type=int)
    parser.add_argument('--epochs', default=45, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')

    # Model parameters
    parser.add_argument('--model', default='deit_small_MCTG', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use Auto Augment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--img-list', default='', type=str, help='image list path')
    parser.add_argument('--data-set', default='', type=str, help='name of dataset')

    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument('--gen_attention_maps', action='store_true')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default=None)
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')

    parser.add_argument('--patch-attn-refine', action='store_true', help='Use patch_attn to refine the cls_attention')
    parser.add_argument('--visualize-cls-attn', action='store_true')

    parser.add_argument('--gt-dir', type=str, default=None)
    parser.add_argument('--cam-npy-dir', type=str, default=None)
    parser.add_argument("--scales", nargs='+', type=float)
    parser.add_argument('--label-file-path', type=str, default=None)
    parser.add_argument('--attention-type', type=str, default='fused')

    parser.add_argument('--out-crf', type=str, default=None)
    parser.add_argument("--low_alpha", default=1, type=int)
    parser.add_argument("--high_alpha", default=12, type=int)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.dist_url = 'env://'
    
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    dist.barrier()
    

def load_model_weight_mctg(args, model):
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else: checkpoint = torch.load(args.finetune, map_location='cpu')

    try: checkpoint_model = checkpoint['model']
    except: checkpoint_model = checkpoint
        
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    Np = int(args.input_size // model.patch_embed.patch_size[0])
    
    if args.finetune.startswith('https'):
        num_extra_tokens = 1
    else:
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    original_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

    if args.finetune.startswith('https'):
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.nb_classes, 1)
    else:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    pos_tokens = pos_tokens.reshape( # (1, Hp, Wp, C)->(1, C, Hp, Wp)
        -1, original_size, original_size, embedding_size).permute(0, 3, 1, 2)
    
    import torch.nn.functional as F
    pos_tokens = F.interpolate(
            input=pos_tokens,
            size=(Np, Np),
            mode='bicubic',
            align_corners=False)
    
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    checkpoint_model['pos_embed_cls'] = extra_tokens
    checkpoint_model['pos_embed_pat'] = pos_tokens

    if args.finetune.startswith('https'):
        cls_token_checkpoint = checkpoint_model['cls_token']
        new_cls_token = cls_token_checkpoint.repeat(1, args.nb_classes, 1)
        checkpoint_model['cls_token'] = new_cls_token
    
    return checkpoint_model
      

def ddp_print(logger, log_msg):
     if dist.get_rank() == 0:
         logger.info(log_msg)
         
         
def main(args):
    session_name = 'Train-classification'
    args = parser.parse_args()
    init_distributed_mode(args)   
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    same_seeds(args.seed)
    
    # Train and Validation for image classification
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(
        args.train_list, 
        voc12_root=args.voc12_root,
        resize_long=(320, 640), 
        hor_flip=True,
        crop_size=512, 
        crop_method="random")
    
    sampler_train = DistributedSampler(train_dataset)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        ampler=sampler_train,
        batch_size=args.batch_per_gpu,
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=True)
    
    val_dataset = voc12.dataloader.VOC12ClassificationDataset(
        args.val_list, 
        voc12_root=args.voc12_root,
        crop_size=512)
    
    sampler_val = DistributedSampler(val_dataset)
     
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_per_gpu),
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=True)
    
    # create model
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        input_size=args.input_size) 
    
    avg_meter = pyutils.AverageMeter()

    bestckpt_name = 'deit_small_MCTG_best.pth'
    output_path = 'voc_mctg'
    
    data_mkdir(output_path)
    checkpoint_path = output_path
    data_mkdir(checkpoint_path)
    
    logger_info(logger_name=session_name, 
                log_path=os.path.join(output_path, 'train_cls.log'))
    logger = logging.getLogger(session_name)
    ddp_print(logger, f"Use seed: {args.seed}")
    
    if args.finetune:
        checkpoint_model = load_model_weight_mctg(args, model)
        model.load_state_dict(checkpoint_model, strict=False)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(logger, f'Number of parameters: {n_parameters}')
    print(logger, bestckpt_name)

    linear_scaled_lr = args.lr * args.batch_per_gpu * dist.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    train_one_epoch = model_train_dict[model_dir]
    
    optimizer = create_optimizer(args, model)
    criterion = nn.MultiLabelSoftMarginLoss()
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    ddp_print(logger, f"Total epochs: {args.epochs}")
    
    # to ddp model
    model.to(device)
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True, device_ids=[args.local_rank])
    
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model, 
            data_loader=data_loader_train,
            optimizer=optimizer, 
            criterion=criterion,
            device=device, 
            epoch=epoch, 
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad)

        lr_scheduler.step(epoch)

        test_stats = evaluate(
            model=model, 
            data_loader=data_loader_val, 
            device=device)
    
        ddp_print(logger, f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        if test_stats["mAP"] > max_accuracy:
            torch.save({'model': model.module.state_dict()}, 
                       os.path.join(checkpoint_path, f'{args.model}_best.pth'))

        max_accuracy = max(max_accuracy, test_stats["mAP"])
        ddp_print(logger, f'Max mAP: {max_accuracy * 100:.2f}%')

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()}}

        if utils.is_main_process():
            ddp_print(logger, json.dumps(log_stats))

    torch.save({'model': model.module.state_dict()}, os.path.join(checkpoint_path, f'{args.model}_last.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    ddp_print(logger, 'Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'classification model training and evaluation script', 
        parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
