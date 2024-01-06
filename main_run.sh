### Train classification network and generate LPCAM seeds.
CUDA_VISIBLE_DEVICES=0,1 \
python run_sample.py \
    --train_cam_pass True \
    --make_cam_pass True \
    --eval_cam_pass True \

# CUDA_VISIBLE_DEVICES=0,1 \
#     python run_sample.py \
#     --make_sem_seg_pass True \
#     --eval_sem_seg_pass True \
