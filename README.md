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
- Elisa Ferrara (SCIPER ID: 371064)

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

## Implemented models
In this branch we implemented the following models:
- PTR
- Our modified version of QCNet

## Code structure
The code is structured as follows:
- ```configs```: In this folder there are the configuration files for the models. The ```config.yaml``` file contains the default configuration settings for the training script. Then in the folder ```method``` there are the configuration files for the models.
- ```dataloader```: In this folder there are the dataloaders for the models.
- ```datasets```: In this folder there are the datasets for the models. Here the post-processing of the data is done together with the data manipulation.
- ```models```: In this folder there are the models implemented.
- ```utils```: In this folder there are the utility functions used in the code.
- ```generate_predictions.py```: This file can be used to generate the predictions for the kaggle competition.
- ```k_means.py```: This file can be used to generate the k-means clustering of the data.
- ```train.py```: This file can be used to train the models.