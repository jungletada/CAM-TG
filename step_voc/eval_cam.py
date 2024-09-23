import os
import numpy as np
from voc12.dataloader import VOCSegmentationLabelDataset
from misc.pyutils import calc_semantic_segmentation_confusion

    
def run(args):
    dataset = VOCSegmentationLabelDataset( 
        data_dir=args.voc12_root,
        id_list_file=args.infer_list)
    
    def eval_curve(threshold):
        preds = []
        labels = []
        miou = 0.
        for i, img_id in enumerate(dataset.ids):
            cam_dict = np.load(os.path.join(args.eval_cam_dir, img_id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res'] # (#val_cls, H, W)
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant') # [0, cls1, ...]
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(dataset[i]["label"])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        miou = np.nanmean(iou)
        print("Threshold: {:.2f}, mIoU: {:.4f}".format(threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    best_res = 0.
    best_threshold = 0
    for t in range(40, 60):
        miou = eval_curve(t / 100.)
        if miou > best_res: 
            best_res = miou
            best_threshold = t / 100.
        else:
            break
            
    print("-"*30)
    print("Best threshold: {}, best mIoU: {:.4f}, num_imgs: {}".format(
        best_threshold, best_res, len(dataset.ids)))