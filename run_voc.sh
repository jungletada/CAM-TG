GPU=0,1
CUDA_VISIBLE_DEVICES=${GPU} \

python run_sample.py \
    --make_cam_pass True \
    --eval_cam_pass True \
    # --infer_list voc12/train.txt \

#>>>>>> make IR label
python run_sample.py \
    --cam_to_ir_label_pass True \
    --conf_fg_thres 0.48 \
    --conf_bg_thres 0.28 \
# --ir_palette_dir ir_palette \

#>>>>>> train IRN
python run_sample.py \
    --train_irn_pass True \

#>>>>>> make segmentation label
python run_sample.py \
    --beta 11 \
    --exp_times 8 \
    --sem_seg_bg_thres 0.5 \
    --make_sem_seg_pass True \
    --eval_sem_seg_pass True \
    --sem_seg_npy_dir sem_seg_npy 
    # --infer_list voc12/train.txt \

# #>>>>>> Test for background threshold
# python run_sample.py \
#     --sem_seg_npy_dir sem_seg_npy \
#     --eval_cam_dir sem_seg_npy \
#     --eval_cam_pass True
    