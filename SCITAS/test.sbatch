#!/bin/bash
#SBATCH --chdir /home/dalbesio/DLAV-project
#SBATCH --ntasks=1
#SBATCH --mem=50000M
#SBATCH --time=00:15:00
#SBATCH --output SCITAS/logs/log_%j.log
#SBATCH --output SCITAS/logs/out_%j.out

# Load the required modules
#module load gcc cuda python

# Activate the virtual environment
source venv/bin/activate

# Move to motionnet
cd motionnet

# Run the script
python train.py