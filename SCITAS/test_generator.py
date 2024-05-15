## Parameters definition ##
data = {
    "seed": [37],
    "past_len": [21],
    "future_len": [60],
    "num_modes": [5],
    "hidden_size": [128],
    "num_encoder_layers": [2],
    "num_decoder_layers": [4],
    "tx_hidden_size": [384],
    "tx_num_heads": [16],
    "dropout": [0.1],
    "max_epochs": [100],
    "learning_rate": [0.0005],
    "learning_rate_scheduler": [[15, 25, 45, 55, 65, 75]],
    "train_batch_size": [256],
    "entropy_weight": [40.0],
    "aug_mode": ["none"],
    "radius": [0.0],
    "prob_car": [0.0],
    "prob_aug": [0.0]
}
# Attention: The file will generate a combination of all the different possibilities.
# For example if you have seed: [1, 2] and past_len: [1, 2], the file will generate 4 experiments:
# seed: 1, past_len: 1
# seed: 1, past_len: 2
# seed: 2, past_len: 1
# seed: 2, past_len: 2
# Be wise when you define the parameters :)

def callback(data):
    # Define the name rule for the experiments (to see it on SCITAS). data contains the parameters of the experiment.
    name = f"test_modes_{data['num_modes']}"

    # Return the name
    return name

########################################################################################

## Import the libraries ##
import os
import numpy as np
import argparse


def main(experiments_folder_name, config_name, method_name):
    ## Parameter definition ##

    # Get the username by the system (Linux or Windows)
    try:
        user = os.environ['USER']
    except KeyError:
        user = os.environ['USERNAME']

    # Delete the experiment folder
    if os.path.exists(experiments_folder_name):
        os.system(f"rm -rf {experiments_folder_name}")

    # Load the parameters
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), experiments_folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define the indices
    indeces = []
    keys = list(data.keys())
    for key in keys:
        indeces.append([i for i in range(len(data[key]))])

    # Define all the combinations
    exp_num = 0
    cart_index = cartesian(indeces)
    print(f"Total number of experiments: {len(cart_index)}")
    for i in cart_index:
        data_exp = {}
        for k in range(len(keys)):
            data_exp[keys[k]] = data[keys[k]][i[k]]    
    
        # Define the name format        
        name = callback(data_exp) if callback is not None else None

        # Get the configuration files
        exp_name = f"{name}" if name is not None else f"exp_{i}"
        config, ptr = get_config_files(user, exp_name, data_exp)

        # Create the directory
        dir_name = f"exp_{exp_num}"
        exp_num += 1
        dir_path = os.path.join(save_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Save the configuration file
        with open(os.path.join(dir_path, f"{config_name}.yaml"), "w") as f:
            f.write(config)
        print(f"Configuration file saved in {dir_path} as {config_name}.yaml")

        # Save the PTR method file
        with open(os.path.join(dir_path, f"{method_name}.yaml"), "w") as f:
            f.write(ptr)
        print(f"PTR method file saved in {dir_path} as {method_name}.yaml")

def cartesian(arrays, out=None):
    """
    Credits: This function has been taken from https://stackoverflow.com/a/1235363
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def get_config_files(user,name,data):

    # Check if all keys are present
    dir_key_needed = [
        'past_len',
        'future_len',
        'num_modes',
        'hidden_size',
        'num_encoder_layers',
        'num_decoder_layers',
        'tx_hidden_size',
        'tx_num_heads',
        'dropout',
        'max_epochs',
        'learning_rate',
        'learning_rate_scheduler',
        'train_batch_size'
    ]
    for key in dir_key_needed:
        if key not in data:
            raise ValueError(f"Key {key} not found in data")

    # Define the config file
    config = f"""
# exp setting
exp_name: '{name}'
ckpt_path: Null
seed: {data['seed']}
debug: False
devices: [0]

# data related
load_num_workers: 0
train_data_path: ['/home/{user}/DLAV-project/.datasets/train']
val_data_path: ['/home/{user}/DLAV-project/.datasets/val']
max_data_num: [1000000]
past_len: {data['past_len']} # 2.1 s
future_len: {data['future_len']} # 6 s
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
num_modes: {data['num_modes']}
hidden_size: {data['hidden_size']}
num_encoder_layers: {data['num_encoder_layers']}
num_decoder_layers: {data['num_decoder_layers']}
tx_hidden_size: {data['tx_hidden_size']}
tx_num_heads: {data['tx_num_heads']}
dropout: {data['dropout']}
entropy_weight: {data['entropy_weight']}
kl_weight: 20.0
use_FDEADE_aux_loss: True

# train
max_epochs: {data['max_epochs']}
learning_rate: {data['learning_rate']}
learning_rate_sched: {data['learning_rate_scheduler']}
optimizer: Adam 
scheduler: multistep
ewc_lambda: 2000
train_batch_size: {data['train_batch_size']}
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

# Set the parameters regarding the standard deviation of the trajectory noise
aug_mode: {data['aug_mode']}
radius: {data['radius']}
prob_car: {data['prob_car']}
prob_aug: {data['prob_aug']}
            """
    return config, ptr

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--experiments_folder_name", type=str, default=None)
    args.add_argument("--config_name", type=str, default=None)
    args.add_argument("--method_name", type=str, default=None)    
    args = args.parse_args()
    main(args.experiments_folder_name, args.config_name, args.method_name)