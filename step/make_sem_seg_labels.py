import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
import importlib

from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import voc12.dataloader
from misc import torchutils, indexing


cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
        64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
        0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
        64,64,0,  192,64,0,  64,192,0, 192,192,0]


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(
        databin, shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            original_size = np.asarray(pack['size'])
            edge, dp = model(pack['img'][0].cuda(non_blocking=True))
            
            cam_dict = np.load(f"{args.lpcam_out_dir}/{img_name}.npy", allow_pickle=True).item()
            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(
<<<<<<< HEAD
                cam_downsized_values, edge, beta=args.beta, 
                exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
=======
                cam_downsized_values, edge, 
                beta=args.beta, 
                exp_times=args.exp_times, 
                radius=5)

            rw_up = F.interpolate(
                input=rw, 
                scale_factor=4, 
                mode='bilinear', 
                align_corners=False)[..., 0, :original_size[0], :original_size[1]]
            
>>>>>>> 404dabd8baa2e6beac496c0353f6fbbbf7b5864f
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

    from voc12.dataloader import VOC12ClassificationDatasetMSF
    dataset = VOC12ClassificationDatasetMSF(
        args.infer_list,
        voc12_root=args.voc12_root,
        scales=(1.0,),
        make_seg=True)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
