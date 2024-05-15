# DLAV Project

This repository contains the code for the project of the course "Deep Learning for Autonomous Vehicles" at EPFL. The project aims to predict future trajectories of agents in a dynamic environment, using a combination of convolutional and transformer-based architectures to process both image-based and point-based map data. It utilizes attention mechanisms to integrate social and temporal contexts, predicting multiple possible future paths by generating trajectory distributions, and calculates associated probabilities for each predicted mode.

## Table of contents
* [Course information](#course-information)
* [Team members](#team-members)
* [Installation](#installation)
* [Source files](#source-files)
* [How to use the code](#how-to-use-the-code)
  * [Grid Search implementation](#grid-search-implementation)

## Course information
* Lecturer: [Alahi Alexandre Massoud][jpt]
* [CIVIL-459 coursebook][coursebook]

[jpt]: https://people.epfl.ch/129343
[coursebook]: https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459

## Team members
This project has been done by (alphabetically ordered):
- Alessandro Dalbesio (SCIPER ID: 359822)
- Elisa Ferrara (SCIPER ID: )

## Installation

First start by cloning the repository:
```bash
git clone https://github.com/alessandrodalbesio/DLAV-project.git
cd DLAV-project
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate.bat` instead

# Install MetaDrive Simulator
pip install git+https://github.com/metadriverse/metadrive.git

# Install ScenarioNet
pip install git+https://github.com/metadriverse/scenarionet.git

```

Finally, install Unitraj and login to wandb via:
```bash
cd DLAV-project # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```

## Main files
### `ptr.py`
This Python script defines a neural network model specifically designed for predicting future trajectories of agents in a dynamic environment, using a combination of convolutional and transformer-based architectures to process both image-based and point-based map data. It utilizes attention mechanisms to integrate social and temporal contexts, predicting multiple possible future paths by generating trajectory distributions, and calculates associated probabilities for each predicted mode.
### `ptr.yaml`
This YAML file contains the default configuration settings for the neural network model, including the number of input channels, hidden dimensions, number of transformer layers, dropout rate, and other architectural parameters. It also specifies the hyperparameters for training, such as the learning rate, batch size, and number of epochs.
### `train.py`
This Python script uses PyTorch Lightning and Hydra to set up and execute the training of a machine learning model, with configurations loaded and managed dynamically via Hydra from a specified directory. It prepares the training and validation datasets, utilizes data loaders with custom batch sizes, and sets up a training loop with model checkpoints based on validation performance, supporting both local and distributed training environments depending on the configuration.
### `config.yaml`
This YAML file contains the default configuration settings for the training script. It also specifies the paths to the training and validation datasets, the directory for saving model checkpoints, and the number of GPUs to use for training. <br>
Here you can also define your optimizer and your scheduler. Currently we support the followings: 
- Optimizer: Adam ("adam"), AdamW ("adamw"), SGD ("sgd") ;
- Scheduler: Multistep ("multistep"), Plateau ("plateau").
Other optimizers or schedulers can be added in the config.yaml file.

### `test_generator.py`
This file can be used to generate multiple configuration files for different hyperparameters. It is particularly useful to automate a grid search process.
### `scheduler.sh`
This file can be used to manage the copy of the configuration files generated with `test_generator.py` in the correct directory and to run the training script for each configuration.
### `run_scheduler.sbatch`
This file can be used to run the grid search process on a cluster.
### `run_submission.sbatch`
This file can be used to run the code needed to generate the submission file for the competition.



## How to use the code
1) Follow the installation instructions above
2) Use a GPU tu train the model. You can find the training and validation dataset at this link: https://drive.google.com/drive/u/0/folders/1ta4Mw09DXrk3MPM4XGwwG79VZzE4NfLJ . Save the dataset and update the path in config.yaml. The default position is `DLAV-project/.datasets/`.
3) Here you have two possibilities:
    - If you are doing a grid search then define the parameters in `test_generator.py`.
    - If you are not doing a grid search then update manually the parameters in `config.yaml` and in `ptr.yaml`.
4) Run the sbatch file on your terminal (i.e. `sbatch run_scheduler.sbatch` if you are doing a grid search or `sbatch run_training.sbatch` if you are not doing a grid search).





