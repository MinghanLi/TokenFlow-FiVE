# # inversion 500 steps
# CUDA_VISIBLE_DEVICES=2 python preprocess.py \
#     --data_dir data \
#     --dataset_json configs/dataset.json 

# # sampling or editing videos
# CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
#     --n_frames 40 \
#     --data_dir data \
#     --data_resize_dir data_resize \
#     --dataset_json configs/dataset.json 

# edit single video
# python run_tokenflow_pnp.py --config_path configs/config_pnp_woman-running.yaml

# FiVE 
CUDA_VISIBLE_DEVICES=2 python preprocess.py \
    --data_dir data_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

# # sampling or editing videos
CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE \
    --data_resize_dir data_FiVE_resize \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE \
    --data_resize_dir data_FiVE_resize \
    --dataset_json data_FiVE/edit_prompt/edit2_FiVE.json 

CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE \
    --data_resize_dir data_FiVE_resize \
    --dataset_json data_FiVE/edit_prompt/edit3_FiVE.json 

CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE \
    --data_resize_dir data_FiVE_resize \
    --dataset_json data_FiVE/edit_prompt/edit4_FiVE.json 