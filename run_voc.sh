GPU=0,1
CUDA_VISIBLE_DEVICES=${GPU} \


# python run_sample_voc.py \
#     --train_cam_pass True \

# python run_sample_voc.py \
#     --make_cam_pass True \


# python run_sample_voc.py \
#     --eval_cam_pass True \
#     --infer_list voc12/train.txt \


# #>>>>>> make IR label
python run_sample_voc.py \
    --cam_to_ir_label_pass True


# #>>>>>> train IRN
# python run_sample_voc.py \
#     --train_irn_pass True


# #>>>>>> make segmentation label
# python run_sample_voc.py \
#     --make_sem_seg_pass True \
#     --eval_sem_seg_pass True \
#     --infer_list voc12/train.txt
