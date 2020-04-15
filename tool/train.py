r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py 
    python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train.py 
python -m torch.distributed.launch --nproc_per_node=1 --use_env tool/train.py 

"""
import datetime
import os
import time
from PIL import Image
from torchvision.transforms import functional as F
import models.transforms as T
import torch
import torch.utils.data
from torch import nn
import sys
from models import mask_rcnn
from models.coco_utils import get_coco_for_apollo

from models.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from models.engine import train_one_epoch, evaluate
from models import utils

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_model_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)
    return model

def get_double_backbone_model(args):
    main = '/media/vrlab/556cb30b-ef30-4e11-a77a-0e33ba901842/Expriments/CVPR_model/Main_backbone/maskrcnn_apollocar3d.pth'
    aux = '/media/vrlab/556cb30b-ef30-4e11-a77a-0e33ba901842/Expriments/part_seg3/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
    model = mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, num_classes=3, main_backbone_pretrained_path=args.main_backbone_path, aux_backbone_pretrained_path=args.aux_backbone_path)
    for name,param in model.named_parameters():
        if "backbone" in name:
            #　print(name)
            param.requires_grad = False
    return model


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset = get_coco_for_apollo(os.path.join(args.data_path, 'result_more_colors'),
     os.path.join(args.data_path, 'cus_editing_data.json'), get_transform(train=True))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    

    print("Creating model")
    model = get_double_backbone_model(args)
    
    model.to(device)
    
            
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    

    print("Start training")
    start_time = time.time()
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, 10)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args},
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='./editing_data', help='dataset')
    parser.add_argument('--main-backbone-path', default='./pretrained_model/main_backbone.pth', help='mainbackbone path')
    parser.add_argument('--aux-backbone-path', default='./pretrained_model/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth', help='auxbackbone path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
   

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)
    # infer_samples()
    main(args)
