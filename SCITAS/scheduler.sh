#!/bin/bash

# Set the username
username="dalbesio"

# Activate the virtual environment
source /home/$username/DLAV-project/.venv/bin/activate

# Set the directory path
sched_config="/home/$username/DLAV-project/SCITAS/sched_config"
dest_dir="/home/$username/DLAV-project/motionnet/configs"

# Move to the main directory
cd /home/$username/DLAV-project/motionnet

if [ -d "$sched_config" ]; then
    for exp_directory in "$sched_config"/*; do
        if [ -d "$exp_directory" ]; then
            echo "Running experiment in directory: $exp_directory"

            # Copy the config files to the main directory
            cp "$exp_directory/config.yaml" "$dest_dir/config.yaml"
            cp "$exp_directory/ptr.yaml" "$dest_dir/method/ptr.yaml"

            echo "Copied config files to the main directory"
            # Run the experiment
            python train.py method=ptr
        fi
    done
else
    echo "Directory does not exist: $directory"
fi