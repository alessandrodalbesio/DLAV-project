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
git clone https://github.com/alessandrodalbesio/DLAV-project.git
cd DLAV-project
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate # .\venv\Scripts\activate.bat

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
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.

## Source files
### ptr.py
Prediction, Trajectory, Representation. 
The Python script defines a neural network model specifically designed for predicting future trajectories of agents in a dynamic environment, using a combination of convolutional and transformer-based architectures to process both image-based and point-based map data. It utilizes attention mechanisms to integrate social and temporal contexts, predicting multiple possible future paths by generating trajectory distributions, and calculates associated probabilities for each predicted mode.
### train.py
This Python script uses PyTorch Lightning and Hydra to set up and execute the training of a machine learning model, with configurations loaded and managed dynamically via Hydra from a specified directory. It prepares the training and validation datasets, utilizes data loaders with custom batch sizes, and sets up a training loop with model checkpoints based on validation performance, supporting both local and distributed training environments depending on the configuration.


## How to use the code
1) Follow the installation instructions above
2) Use a GPU tu train the model. You can find the training and validation dataset at this link: https://drive.google.com/drive/u/0/folders/1ta4Mw09DXrk3MPM4XGwwG79VZzE4NfLJ . Save the dataset and update the path in config.yaml
3) Update config.yaml and ptr.yaml with the desired hyperparameters, variable names and data paths.
4) Run the sbatch file on your terminal

### Grid Search implementation
In order to look for the best parameters, execute the file run_scheduler.sbatch. It will call both test_generator.py and scheduler.sh :
*scheduler.sh* :
This Bash script activates a Python virtual environment, iterates through a directory of experiment configurations, copies specific configuration files into a designated directory, and then executes a training script for each configuration using PyTorch Lightning.
*test_generator.py*:
This Python script generates a series of configuration files for machine learning experiments, varying parameters like learning rate, batch size, and learning rate scheduler intervals, and saves these configurations into a structured directory format for subsequent execution.







