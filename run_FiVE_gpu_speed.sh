# FiVE 
CUDA_VISIBLE_DEVICES=1 python preprocess.py \
    --data_dir data_FiVE/images \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 

CUDA_VISIBLE_DEVICES=1 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data_FiVE/images \
    --data_resize_dir data_FiVE_resize \
    --output_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json 