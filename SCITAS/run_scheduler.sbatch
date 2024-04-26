#!/bin/bash
#SBATCH --chdir /home/dalbesio/DLAV-project/SCITAS
#SBATCH --ntasks=1
#SBATCH --account=civil-459
#SBATCH --qos=civil-459
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=24:00:00
#SBATCH --output logs/log_%j.log
#SBATCH --output logs/out_%j.out

# Run the python file
python test_generator.py

# Run the batch file 
bash scheduler.sh