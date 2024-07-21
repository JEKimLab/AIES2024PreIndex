# Estimating Environmental Cost Throughout Model's Adaptive Life Cycle

This repository contains all the necessary code to reproduce the research work to obtain PreIndex for two data distributions when retraining a model.

## Installation

To install all the required dependencies and python version in an Ubuntu environment, execute the following command:

```bash
sudo apt update
sudo apt install python3.8
pip install -r requirements.txt
```
<br/>


## PreIndex
To obtain PreIndex for distributional shift:
```bash
python3.8 PreIndex/pre_index.py -m MODEL_PATH/model.pkl -s SAVE_RESULTS_TO -d DATASET -cl LAYER_NAME -rs RANDOM_SEED -mt MODEL_TYPE -n_t NOISE_TYPE -n_l LEVEL 
```
Sample command:
```bash
python3.8 PreIndex/pre_indext.py -m ResNet18/model.pkl -s ResNet18/Sample -d cifar10 -cl layer4.1.conv2 -rs 1 -mt cnn -n_t gauss -n_l 0.05

```
<br/>


## Retraining
### To retrain a model, execute `retrain_dir/retrain_task.py` and `retrain_dir/retrain_dist.py` script for task change and distribution shift, respectively, in the following format:
```bash
python3.8 retrain_dir/retrain_dist.py -mp PATH_TO_MODEL/model.pkl -save PATH_TO_SAVE -acc CUTOFF_ACCURACY -rs RANDOM_SEED -tlc TRANSFORMS_LR_CUTOFF -d DATASET -n_tp NOISE_TYPE -n_lvl LEVEL
```
```bash
python3.8 retrain_dir/retrain_task.py -mp PATH_TO_MODEL/model.pkl -save PATH_TO_SAVE -acc CUTOFF_ACCURACY -rs RANDOM_SEED -tl TRANSFORMS_LR -d DATASET
```
## Code Carbon Initialization
To initialize Code Carbon, for measuring energy and carbon emission when executing `retrain_dir/retrain.py`, run the following command:
```bash
! codecarbon init
```
