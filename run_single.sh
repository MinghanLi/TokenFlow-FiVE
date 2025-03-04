# inversion 500 steps
CUDA_VISIBLE_DEVICES=2 python preprocess.py \
    --data_dir data \
    --dataset_json configs/dataset.json 

# sampling or editing videos
CUDA_VISIBLE_DEVICES=2 python run_tokenflow_pnp.py \
    --n_frames 40 \
    --data_dir data \
    --data_resize_dir data_resize \
    --dataset_json configs/dataset.json 

# edit single video
python run_tokenflow_pnp.py --config_path configs/config_pnp_woman-running.yaml