# python run_sample.py \
#     --make_cam_pass True \
#     --eval_cam_pass True \


python run_sample.py \
--conf_fg_thres 0.48 \
--conf_bg_thres 0.26 \
--sem_seg_bg_thres 0.49 \
--cam_to_ir_label_pass True \
--ir_palatte_dir ir_palette \
--train_irn_pass True \
--make_sem_seg_pass True \
--eval_sem_seg_pass True \
    

# python run_sample.py \
#     --sem_seg_npy_dir sem_seg_npy \
#     --eval_cam_pass True \
#     --eval_cam_dir sem_seg_npy \
    