# FiVE 
CUDA_VISIBLE_DEVICES=0 python preprocess.py \
    --data_dir data_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

# # # sampling or editing videos
# CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
#     --n_frames 40 \
#     --data_dir data_FiVE \
#     --data_resize_dir data_FiVE_resize \
#     --output_dir outputs_FiVE \
#     --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

# CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
#     --n_frames 40 \
#     --data_dir data_FiVE \
#     --data_resize_dir data_FiVE_resize \
#     --dataset_json data_FiVE/edit_prompt/edit2_FiVE.json 

# CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
#     --n_frames 40 \
#     --data_dir data_FiVE \
#     --data_resize_dir data_FiVE_resize \
#     --dataset_json data_FiVE/edit_prompt/edit3_FiVE.json 

# CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
#     --n_frames 40 \
#     --data_dir data_FiVE \
#     --data_resize_dir data_FiVE_resize \
#     --dataset_json data_FiVE/edit_prompt/edit4_FiVE.json 