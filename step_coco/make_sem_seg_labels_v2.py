import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import os
import imageio
import importlib
import numpy as np
from tqdm import tqdm
import os.path as osp
import mscoco.dataloader
from misc import torchutils, indexing, imutils


cudnn.enabled = True

def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(
        databin,
        shuffle=False,
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = pack['name'][0].split('.')[0]
            
            if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
                continue
            
            img_size = pack['size']
            edge, _ = model(pack['img'][0].cuda(non_blocking=True))
            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
            
            keys = []
            cams = []
            for cls, cam_map in cam_dict.items():
                keys.append(cls)
                cams.append(cam_map)
                
            # cams = np.power(cam_dict['cam'], 1.5) # Anti
            keys = np.pad(np.array(keys) + 1, (1, 0), mode='constant')
            
            if keys.shape[0] == 1:
                conf = np.zeros_like(pack['img'][0])[0, 0]
                imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), conf.astype(np.uint8))
                continue
            
            cams = np.concatenate(
                [np.expand_dims(cam_map, axis=0) for cam_map in cams], axis=0)
            strided_size = imutils.get_strided_size(img_size, 4)
            cams = torch.from_numpy(cams).unsqueeze(1) # Cls, 1, H//4, W//4
            cams = F.interpolate(cams, strided_size, mode='bilinear', align_corners=False)
            cam_downsized_values = cams.squeeze(1).cuda()
            
            rw = indexing.propagate_to_edge(
                cam_downsized_values, 
                edge, 
                beta=args.beta, 
                exp_times=args.exp_times, 
                radius=5)
            
            rw_up = F.interpolate(
                rw, 
                img_size, 
                mode='bilinear', 
                align_corners=False)
            
            rw_up = rw_up.squeeze(1)
            rw_up = rw_up / torch.max(rw_up)
            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = mscoco.dataloader.COCOClassificationDatasetMSF(
        image_dir = osp.join(args.mscoco_root,'train2014/'),
        anno_path= osp.join(args.mscoco_root,'annotations/instances_train2014.json'),
        labels_path='./mscoco/train_labels.npy', 
        scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
