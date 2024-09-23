import os
import imageio
import numpy as np
from voc12.dataloader import VOCSegmentationLabelDataset
from misc.pyutils import calc_semantic_segmentation_confusion


def run(args):
    dataset = VOCSegmentationLabelDataset(
        data_dir=args.voc12_root,
        id_list_file=args.infer_list)
    preds = []
    labels = []
    n_img = 0
    for i, id in enumerate(dataset.ids):
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        labels.append(dataset[i]["label"])
        n_img += 1

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("Total images", n_img)
    print("False Positive: {:.4f}, False Negative: {:.4f}".format(fp[0], fn[0]))
    print("IoU (%) for each class:")
    for res in iou:
        print("{:.2f}".format(res * 100.))
    print("mIoU (%): {:.2f}".format(np.nanmean(iou) * 100.))
