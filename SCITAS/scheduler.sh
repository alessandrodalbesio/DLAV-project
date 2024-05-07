#!/bin/bash

# Define some variables
experiments_folder_name="experiments"
name_method_file="ptr"
name_config_file="config"

# Get the username
username=$(whoami)
echo "Running the scheduler script as user: $username"

# Activate the virtual environment
source /home/$username/DLAV-project/.venv/bin/activate

# Set the directory path
sched_config="/home/$username/DLAV-project/SCITAS/$experiments_folder_name"
dest_dir="/home/$username/DLAV-project/motionnet/configs"

# Run the test generator script
python /home/$username/DLAV-project/SCITAS/test_generator.py --experiments_folder_name $experiments_folder_name --method_name $name_method_file --config_name $name_config_file

# Move to the main directory
cd /home/$username/DLAV-project/motionnet

if [ -d "$sched_config" ]; then
    for exp_directory in "$sched_config"/*; do
        if [ -d "$exp_directory" ]; then
            echo "Running experiment in directory: $exp_directory"

            # Copy the config files to the main directory
            cp "$exp_directory/$name_config_file.yaml" "$dest_dir/$name_config_file.yaml"
            cp "$exp_directory/$name_method_file.yaml" "$dest_dir/method/$name_method_file.yaml"

            echo "Copied config files to the main directory"

            # Run the experiment
            python train.py method=ptr
        fi
    done
else
    echo "Directory does not exist: $directory"
fi