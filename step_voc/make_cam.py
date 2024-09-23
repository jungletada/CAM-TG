import os
import torch
import importlib
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
from torch.backends import cudnn
from torch import multiprocessing, cuda

import voc12.dataloader
from misc import torchutils, imutils
from torch.utils.data import DataLoader
cudnn.enabled = True
import warnings
warnings.filterwarnings("ignore")


def normalize_cam(cam_mask):
    for i in range(cam_mask.size(0)):
        channel = cam_mask[i]
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        cam_mask[i] = (channel - min_val) / (max_val - min_val + 1e-8)
    
    return cam_mask


def flip_cam(cam_list):
    for i in range(len(cam_list)):
        cam_scale = cam_list[i]
        group1, group2 = cam_scale[0], cam_scale[1]
        group2_flipped = torch.flip(group2, dims=[2])
        cam_list[i] = torch.stack([group1, group2_flipped])
        
    cam_list = [torch.sum(cam, dim=0) for cam in cam_list]
    return cam_list


def rescale_cam(cam_mask):
    threshold = 0.35
    decay_factor = 0.5
    adjusted_cam = torch.where(cam_mask > threshold, cam_mask, cam_mask * cam_mask)
    adjusted_cam = torch.clamp(adjusted_cam, min=0.0, max=1.0)
    return adjusted_cam


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            # batch size = 1, so we only take the first element
            img_name = pack['name'][0] 
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)# floor(W'/4 x H'/4)
            # strided_up_size = imutils.get_strided_up_size(size, 16) # floor(W^ x H^)
            
            outputs = [model.forward(img[0].cuda(non_blocking=True)) # img[0]->[(2, 3, W', H')]
                       for img in pack['img']] # outputs->list[(2, 20, W/16, H/16)]
            #==========strided 4 cam list====================================================#
            strided_cam_list = [# upsample all multi-scale CAMs to strided_size: (W'/4 x H'/4)
                F.interpolate(cam, strided_size, mode='bilinear', align_corners=False)
                for cam in outputs]
            strided_cam_list = flip_cam(strided_cam_list)
            strided_cam = torch.sum(torch.stack(strided_cam_list), 0) # (20, W'/4, H'/4)
            
            #======== high resolution cam list ===============================================#
            highres_cam_list = [# upsample all multi-scale CAMs to strided_up_size->floor(W^, H^)
                F.interpolate(cam, size, mode='bilinear', align_corners=False)
                for cam in outputs] # ->[(2, 20, W, H)
            highres_cam_list = flip_cam(highres_cam_list)
            highres_cam = torch.sum(torch.stack(highres_cam_list, 0), 0) # (20, W, H)
            
            valid_cat = torch.nonzero(label)[:, 0] # get validate class->[#val_cls]
            
            strided_cam = strided_cam[valid_cat]
            strided_cam = normalize_cam(strided_cam)
            
            highres_cam = highres_cam[valid_cat]
            highres_cam = normalize_cam(highres_cam)
            
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()}) 
             
            # cam_dict = {}
            # highres_cam = highres_cam.cpu().numpy()
            # for i, cls in enumerate(valid_cat):
            #     cam_dict[cls] = highres_cam[i]
                
            # np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg', 'npy')),
            #         cam_dict)  
            
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
                
                                    
def run(args):
    
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(
        args.infer_list, 
        voc12_root=args.voc12_root, 
        scales=args.cam_scales) # Trainset-1464
   
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()