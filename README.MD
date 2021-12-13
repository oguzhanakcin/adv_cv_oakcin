# Data Sharing and Data Valuation for Heteregenous Models

This repository contains the code implementations of the experiments of the Advanced Topics in Computer Vision Course ECE381V. 

## How to Use

### Downloading datasets and setting up the environment 

You can download the RoadNet and Multi-Class Weather Prediction Datasets from the link given below.

https://drive.google.com/drive/folders/17H6YFLOaj49KbXzmhmrrCowB5XsS1MuY?usp=sharing

In this folder please download corresponding RoadNet and Multi-Class Weather Prediction zip files containing the datasets. 

Place the datasets in `
./experiments/WeatherPred/data`
 and `
./experiments/RoadNet/data`
 respectively or specify the relative locations of the dataset in the simulations.

In order to create the environment please use the 'environment.yml' file.
Once these steps are done you are ready for experiments

### Running Experiments

The hyperparameters for the experiments are given in the each simulation relative folder hyp.yaml file. This file contains number of rounds, the exact datasets, cache and observation sizes, etc.

After training models the results can be found in the relative folder `./experiments/*/sim_data`, the trained model in the initial state of the model can be found in `./experiments/*/model_weights` folder. In order to change that location please check the corresponding run_simulation.py file, output location can be given as an argument. 

Additionally, if the dataset files are not in the `./experiments/*/data` file then relative location can be given as argument    
#### Synthetic Dataset

In order to run the experiments with this dataset please run the command given below in folder ./experiments/synthetic

```
python3 run_simulation.py --sim-oracle --sim-random --sim-soft --sim-entropy --sim-gu --train-base --create-dataset

```

#### MNIST Dataset

In order to run the experiments with this dataset please run the command given below in folder ./experiments/synthetic

```
python3 run_simulation.py --sim-oracle --sim-random --sim-soft --sim-entropy --sim-gu --train-base 

```

#### Multi Class Weather Prediction Dataset

In order to run the experiments with this dataset please run the command given below in folder ./experiments/Weather

```
python3 run_simulation.py --sim-oracle --sim-random --sim-soft --sim-entropy --sim-gu --train-base 

```

#### RoadNet Dataset

In order to run the experiments with this dataset please run the command given below in folder ./experiments/RoadNet

```
python3 run_simulation.py --sim-oracle --sim-random --sim-soft --sim-entropy --sim-gu --train-base 

```

## Displaying Results

In order to create the graph please tun 'generate_graphs.py' file with giving argument as to dataset location. Example can be given as

```
python3 generate_graphs.py --data-loc ./experiments/MNIST/sim_data

```

