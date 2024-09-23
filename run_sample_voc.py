import os
import argparse
import os.path as osp
from misc import pyutils
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--voc12_root", default='dataset/VOCdevkit/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.1, type=float)
    parser.add_argument("--cam_scales", default=(1.0,),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.45, type=float)
    parser.add_argument("--conf_bg_thres", default=0.25, type=float)
    
    parser.add_argument("--fg_alpha", default=1, type=float)
    parser.add_argument("--bg_alpha", default=3, type=float)
    
    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=11, type=int)
    parser.add_argument("--exp_times", default=8, type=int,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.45, type=float)

    # Output Path
    parser.add_argument("--work_space", default="results_voc/resnet50", type=str)
    parser.add_argument("--log_name", default="info", type=str)
    parser.add_argument("--cam_weights_name", default="resnet50_s16_best.pth", type=str)
    parser.add_argument("--irn_weights_name", default="res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--lpcam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg", type=str)
    
    parser.add_argument("--ir_palette_dir", default=None, type=str)
    parser.add_argument("--eval_cam_dir", default="cam_mask", type=str)
    parser.add_argument("--sem_seg_npy_dir", default="cam_mask", type=str)
    
    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_lpcam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--train_irn_pass", type=str2bool, default=False)
    parser.add_argument("--make_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=False) 
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=False)

    args = parser.parse_args()
    
    args.log_name = osp.join(args.work_space, args.log_name)
    args.cam_weights_name = osp.join(args.work_space, args.cam_weights_name)
    args.irn_weights_name = osp.join(args.work_space, args.irn_weights_name)
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.lpcam_out_dir = osp.join(args.work_space, args.lpcam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space, args.ir_label_out_dir)
    args.sem_seg_out_dir = osp.join(args.work_space, args.sem_seg_out_dir)
    
    args.eval_cam_dir = osp.join(args.work_space, args.eval_cam_dir)
    args.sem_seg_npy_dir = osp.join(args.work_space, args.sem_seg_npy_dir)
    if args.ir_palette_dir is not None:
        args.ir_palette_dir = osp.join(args.work_space, args.ir_palette_dir)
        os.makedirs(args.ir_palette_dir, exist_ok=True)
        
    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.lpcam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    
    os.makedirs(args.eval_cam_dir, exist_ok=True)
    os.makedirs(args.sem_seg_npy_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    
    def print_info(args):
        import pprint
        args_dict = vars(args)
        pprint.pprint(args_dict)
   
    print_info(args)
    
    if args.train_cam_pass is True:
        import step_voc.train_cam
        timer = pyutils.Timer('step_voc.train_cam:')
        step_voc.train_cam.run(args)

    if args.make_cam_pass is True:
        import step_voc.make_cam
        timer = pyutils.Timer('step_voc.make_cam:')
        step_voc.make_cam.run(args)
    
    if args.make_lpcam_pass is True:
        import step_voc.make_lpcam
        timer = pyutils.Timer('step_voc.make_lpcam:')
        step_voc.make_lpcam.run(args)

    if args.eval_cam_pass is True:
        import step_voc.eval_cam
        timer = pyutils.Timer('step_voc.eval_cam:')
        step_voc.eval_cam.run(args)
        
    if args.cam_to_ir_label_pass is True:
        import step_voc.cam_to_ir_label
        timer = pyutils.Timer('step_voc.cam_to_ir_label:')
        step_voc.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step_voc.train_irn
        timer = pyutils.Timer('step_voc.train_irn:')
        step_voc.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step_voc.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step_voc.make_sem_seg_labels:')
        step_voc.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step_voc.eval_sem_seg
        timer = pyutils.Timer('step_voc.eval_sem_seg:')
        step_voc.eval_sem_seg.run(args)

