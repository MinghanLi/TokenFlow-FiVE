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
# Black swan --> Flamingo
# python preprocess.py --data_path data/blackswan --data_path samples/davis/blackswan 
# python run_tokenflow_pnp.py --config_path configs/config_pnp_blackswan_flamingo.yaml
# python run_tokenflow_pnp.py --config_path configs/config_pnp_blackswan_duck.yaml

# # Wolf --> fox
# python preprocess.py --data_path data/wolf
# python run_tokenflow_pnp.py --config_path configs/config_pnp_wolf_fox.yaml

# # Wolf --> bear
# python preprocess.py --data_path data/bear
# python run_tokenflow_pnp.py --config_path configs/config_pnp_wolf_bear.yaml

# Black dress --> Red dress
# python preprocess.py --data_path data/0011_lucia
python run_tokenflow_pnp.py --config_path configs/config_pnp_0011_lucia_man.yaml
python run_tokenflow_pnp.py --config_path configs/config_pnp_0011_lucia_lion.yaml
# python run_tokenflow_pnp.py --config_path configs/config_pnp_0011_lucia_red.yaml