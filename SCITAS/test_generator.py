import os

## User definition ##
user = "dalbesio"

## Parameter definition ##
lr_list = [0.0005]
train_batch_list = [256]
lr_scheduler_list = [[15,25,45,55,65,75]]
dest_folder = f"/home/{user}/DLAV-project/sched_config"
if os.path.exists(dest_folder):
    os.system(f"rm -r {dest_folder}")

## Files definition ##
index = 0
os.makedirs(dest_folder, exist_ok=True)
for lr in lr_list:
    for train_size in train_batch_list:
        for lr_sched in lr_scheduler_list:
            name = f'{lr},{lr_sched},{train_size}'

            config = f"""
# exp setting
exp_name: '{name}'
ckpt_path: Null
seed: 37
debug: False
devices: [0]

# data related
load_num_workers: 0
train_data_path: ['/home/{user}/DLAV-project/.datasets/train']
val_data_path: ['/home/{user}/DLAV-project/.datasets/val']
max_data_num: [1000000]
past_len: 21 # 0.1s
future_len: 60 # 0.1s
object_type: ['VEHICLE'] #, 'PEDESTRIAN', 'CYCLIST']
line_type: ['lane','stop_sign','road_edge','road_line','crosswalk','speed_bump'] #['lane','stop_sign','road_edge','road_line','crosswalk','speed_bump']
masked_attributes: ['z_axis, size'] # 'z_axis, size, velocity, acceleration, heading'
trajectory_sample_interval: 1 # 0.1s
only_train_on_ego: False
center_offset_of_map: [30.0, 0.0]
use_cache: False
overwrite_cache: False
store_data_in_memory: False

# official evaluation
nuscenes_dataroot: '/mnt/nas3_rcp_enac_u0900_vita_scratch/datasets/Prediction-Dataset/nuscenes/nuscenes_root'
eval_nuscenes: False
eval_waymo: False

defaults:
- method: ptr
            """

            ptr = f"""
# common
model_name: ptr
num_modes: 6
hidden_size: 128
num_encoder_layers: 2
num_decoder_layers: 2
tx_hidden_size: 384
tx_num_heads: 16
dropout: 0.1
entropy_weight: 40.0
kl_weight: 20.0
use_FDEADE_aux_loss: True

# train
max_epochs: 80
learning_rate: {lr}
learning_rate_sched: {lr_sched}
optimizer: Adam 
scheduler: multistep 
ewc_lambda: 2000
train_batch_size: {train_size}
eval_batch_size: 256
grad_clip_norm: 5

# data related
max_num_agents: 15
map_range: 100
max_num_roads: 256
max_points_per_lane: 20 
manually_split_lane: False
point_sampled_interval: 1
num_points_each_polyline: 20
vector_break_dist_thresh: 1.0
            """

            # Create a folder for the experiment
            folder = f"{dest_folder}/exp_{index}"
            os.makedirs(folder, exist_ok=True)
            index += 1

            # Write the config file
            with open(f"{folder}/config.yaml", "w") as f:
                # Remove the first line break
                config = config[1:]
                # Write the config file
                f.write(config)

            with open(f"{folder}/ptr.yaml", "w") as f:
                # Remove the first line break
                ptr = ptr[1:]
                # Write the config file
                f.write(ptr)