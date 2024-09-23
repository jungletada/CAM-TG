GPU=0,1
CUDA_VISIBLE_DEVICES=${GPU} \


# #>>>>>> Train classification network and generate LPCAM seeds.
# python run_sample_coco.py \
#     --train_cam_pass True \
#     --make_cam_pass True \
#     --eval_cam_pass True 


#>>>>>> Train IRN and generate pseudo masks.
# python run_sample_coco.py \
    # --cam_to_ir_label_pass True\
    # --train_irn_pass True \
    

python run_sample_coco.py \
    --make_sem_seg_pass True \
    --eval_sem_seg_pass True 

