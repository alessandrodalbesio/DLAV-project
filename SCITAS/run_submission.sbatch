#!/bin/bash
#SBATCH --chdir /home/dalbesio/DLAV-project
#SBATCH --ntasks=1
#SBATCH --account=civil-459
#SBATCH --qos=civil-459
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=00:30:00
#SBATCH --output logs/log_%j.log
#SBATCH --output logs/out_%j.out

# Activate the virtual environment and run the script
source ~/DLAV-project/.venv/bin/activate

# Move to the directory where the script is located
cd motionnet

# Run the script
python generate_predictions.py

# Deactivate the virtual environment
deactivate