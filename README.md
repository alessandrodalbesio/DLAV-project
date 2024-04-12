# DLAV Project

## Course information
* Lecturer: [Alahi Alexandre Massoud][jpt]
* [CIVIL-459 coursebook][coursebook]

[jpt]: https://people.epfl.ch/129343
[coursebook]: https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459

## Team members
This project has been done by (alphabetically ordered):
- Alessandro Dalbesio
- Elisa Ferrara

## Installation

First start by cloning the repository:
```bash
git clone https://github.com/vita-epfl/DLAV-project.git
cd DLAV-project
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e.

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

Finally, install Unitraj and login to wandb via:
```bash
cd DLAV-project # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.


You can verify the installation via running the training script:
```bash
python train.py method=ptr
```
The incomplete PTR model will be trained on several samples of data available in `motionnet/data_samples`.
