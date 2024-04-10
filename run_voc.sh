GPU=0,1
CUDA_VISIBLE_DEVICES=${GPU} \
WORKDIR=results_voc/resnet50


python run_sample.py \
    --work_space ${WORKDIR} \
    --train_cam_pass True \

# python run_sample.py \
#     --work_space ${WORKDIR} \
#     --make_cam_pass True \


# python run_sample.py \
#     --work_space ${WORKDIR} \
#     --eval_cam_pass True \
#     --infer_list voc12/train.txt \

# #>>>>>> make IR label
# python run_sample.py \
#     --work_space ${WORKDIR} \
#     --cam_to_ir_label_pass True \
#     --conf_fg_thres 0.45 \
#     --conf_bg_thres 0.25 \
# # # --ir_palette_dir ir_palette \

# #>>>>>> train IRN
# python run_sample.py \
#     --work_space ${WORKDIR} \
#     --train_irn_pass True \

# #>>>>>> make segmentation label
# python run_sample.py \
#     --work_space ${WORKDIR} \
#     --sem_seg_bg_thres 0.39 \
#     --make_sem_seg_pass True \
#     --eval_sem_seg_pass True \
#     --sem_seg_npy_dir sem_seg_npy 
    # --infer_list voc12/train.txt \

# #>>>>>> Test for background threshold
# python run_sample.py \
#     --sem_seg_npy_dir sem_seg_npy \
#     --eval_cam_dir sem_seg_npy \
#     --eval_cam_pass True
    