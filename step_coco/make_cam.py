import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import os
import importlib
import numpy as np
import os.path as osp
from tqdm import tqdm

import mscoco.dataloader
from misc import torchutils, imutils
cudnn.enabled = True


def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            valid_cat = torch.nonzero(label)[:, 0]
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            if os.path.exists(os.path.join(args.cam_out_dir, img_name.replace('jpg', 'npy'))):
                continue
            
            if valid_cat.shape[0]==0:
                np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                        {"keys": valid_cat,
                         "cam": torch.zeros((0,strided_size[0],strided_size[1])),
                         "high_res": torch.zeros((0,strided_up_size[0],strided_up_size[1]))[:,:size[0],:size[1]]})
                continue
            
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(
                    torch.unsqueeze(o, 0), 
                    strided_size, 
                    mode='bilinear', 
                    align_corners=False)[0] 
                 for o in outputs]), 0)

            highres_cam = [F.interpolate(
                torch.unsqueeze(o, 1), 
                strided_up_size,
                mode='bilinear', 
                align_corners=False) 
            for o in outputs]
            
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=80)
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = mscoco.dataloader.COCOClassificationDatasetMSF(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy', 
        scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
