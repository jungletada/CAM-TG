import os
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


def run(args):
    dataset = VOCSemanticSegmentationDataset(
        split=args.chainer_eval_set, 
        data_dir=args.voc12_root)

    preds = []
    labels = []
    n_images = 0
    for i, img_id in enumerate(dataset.ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.lpcam_out_dir, img_id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res'] # (#val_cls, H, W)
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant') # [0, cls1, ...]
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print("threshold:", args.cam_eval_thres, '\nmiou:', np.nanmean(iou), "\ni_imgs", n_images)
<<<<<<< HEAD
    print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
=======
    print('among_pred fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
>>>>>>> 404dabd8baa2e6beac496c0353f6fbbbf7b5864f

    return np.nanmean(iou)