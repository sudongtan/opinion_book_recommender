# Preference-inconsistent recommenders
## CS

# 1 Introduction
The model has two functions, for a given user:
-

# 2 Environment

## Environment to run the final 
```
conda create --name rs python=3.6.8
conda activate rs
pip install -r requirement_test.txt
```

## Development environment
Note: this environment requires GPU. The cuda version of the local machine might have conflicts with tensorflow. This version has been tested with cuda 9.0.

```
conda create --name dev python=3.6.8
conda activate dev
pip install -r requirement_dev.txt
```

# 3 Code Structure

## /

- final project
```
conda activate rs
jupyter notebook
```
- app.py
- recommend.py: the script called by the notebook 
- train.py: scripts to conduct grid search on model hyperparameters, the full experiments took >100 hrs on a machine with 2 gpus. Currently the epoch is set to 2 and only 1 experiment will be conducted for demonstration purposes. To run the experiment,

```
conda activate dev
python train.py
```

## notebooks: data preprocessing and result analysis
|Notebook name                |Required data folder    |Time to run                   |
|---|---|---|
|1data_preprocessing_filtering|data_raw| 10 min|
|2data_preprocessing_clearning|data_raw|3 min|
|3data_preprocessing_dataset|data_raw|20 min|
|4features_engineering|data_raw|20 - 50 min, depending on gpu and internet speed, requires very powerful gpu|
|5features_topic|data_raw|2 min|
|6result_analysis|results|< 1 min|


## src: code used in development

- chainRec.py: the model constructor
- dataset.py: train/val/test split and sampling


## Others
 - data_raw/
 - data: data required to run the recommender 
 - model: model required to run the recommender
 - results: 

## 
- original dataset, 
- intermediate data saved from data preprocessing and feature extraction



# Reference

https://github.com/huggingface/pytorch-transformers

