import os
import numpy as np
import imageio
from tqdm import tqdm
from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
from PIL import Image


palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
        64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
        0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
        64,64,0,  192,64,0,  64,192,0, 192,192,0,  255,255,255]


def _work(process_id, infer_dataset, args):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(tqdm(infer_data_loader, position=process_id, desc=f'[PID{process_id}]')):
        img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant') # class labels

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), 
                             mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        # pred = imutils.crf_with_alpha(img, cams, alpha=args.fg_alpha, n_labels=keys.shape[0])
        fg_conf = keys[pred]
        
        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), 
                             mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        # pred = imutils.crf_with_alpha(img, cams, alpha=args.bg_alpha, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255            # we do not care
        conf[bg_conf + fg_conf == 0] = 0    # confident background pixels

        imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'), conf.astype(np.uint8))
        
        # put palette
        if args.ir_palette_dir is not None:
            conf[conf == 255] = 28
            ir_palette = conf.astype(np.uint8)
            ir_palette = Image.fromarray(ir_palette, mode='P')
            ir_palette.putpalette(palette)
            ir_palette.save(os.path.join(args.ir_palette_dir, img_name + '.png'))
        
        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    dataset = voc12.dataloader.VOC12ImageDataset(
        args.train_list, voc12_root=args.voc12_root, img_normal=None, to_torch=False)
    print(f"Total images: {len(dataset)}")
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')
