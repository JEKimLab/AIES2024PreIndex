# Estimating Environmental Cost Throughout Model's Adaptive Life Cycle

This is the official implementation of **Estimating Environmental Cost Throughout Model's Adaptive Life Cycle**

## Installation
All library versions are within `requirements.txt`. To install all the required dependencies and python version in an Ubuntu environment, execute the following command:

```bash
sudo apt update
sudo apt install python3.8
pip install -r requirements.txt
```

## PreIndex
To obtain PreIndex for distributional shift:
```bash
python3.8 PreIndex/pre_index.py
  -m MODEL_PATH/model.pkl \
  -s SAVE_RESULTS_TO \
  -d DATASET \
  -cl LAYER_NAME \
  -rs RANDOM_SEED \
  -mt MODEL_TYPE \
  -n_t NOISE_TYPE \
  -n_l LEVEL 
```
Sample command:
```bash
python3.8 PreIndex/pre_index.py
  -m ResNet18/model.pkl \
  -s ResNet18/Sample \
  -d cifar10 \
  -cl layer4.1.conv2 \
  -rs 1 \
  -mt cnn \
  -n_t gauss \
  -n_l 0.05

```

## Retraining
To retrain a model, run `retrain_dir/retrain_dist.py` in the following format with the path of the original model:
```bash
python3.8 retrain_dir/retrain_dist.py
  -mp PATH_TO_MODEL/model.pkl \
  -save PATH_TO_SAVE \
  -acc CUTOFF_ACCURACY \
  -rs RANDOM_SEED \
  -tlc TRANSFORMS_LR_CUTOFF \
  -d DATASET \
  -n_tp NOISE_TYPE \
  -n_lvl LEVEL
```

## Retraining configurations
`options` contains the JSON configurations format for learning rate schedule, training transformations, and cutoff plans. Edit the default values of 0.0 according to the desired training/testing scheme. Test transformations are within `retrain/retrain_dist.py`, which can be edited based on the dataset in use.

## Code Carbon Initialization
To initialize Code Carbon, for measuring energy and carbon emission when executing `retrain_dir/retrain.py`, run the following command to setup the carbon tracker instance:
```bash
! codecarbon init
```

## Citation
```
@inproceedings{sangarya2024estimatingenvironmentalcostmodels,
      title={Estimating Environmental Cost Throughout Model's Adaptive Life Cycle}, 
      author={Vishwesh Sangarya and Richard Bradford and Jung-Eun Kim},
      year={2024},
      booktitle={AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2024}
      url={https://arxiv.org/abs/2408.01446}, 
}
```
