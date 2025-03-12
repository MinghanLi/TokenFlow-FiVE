 #!/bin/bash

# while true; do

# # sampling or editing videos
CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit2_FiVE.json 

CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit3_FiVE.json 

CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit4_FiVE.json 

CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit5_FiVE.json


CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE/images \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit6_FiVE.json 

#     sleep 600
# done