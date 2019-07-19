# Preference-inconsistent recommenders
CS6460

# 1 Introduction
The model has two functions, for a given user:


# 2 Recommend Environment

## 2.1 Environment setup
### Method 1: 
### Method 2:
```
conda create --name rs python=3.6.8
conda activate rs
pip install -r requirements.txt
```
## 2.2 Instructions
|File name                |Required data folder    |Notes                   |
|---|---|---|
|final_project.|---|---|
|analysis_|---|---|
|analysis_|---|---|
|app.py|---|---|

## 2.3 Source code
|File name                |Required data folder    |Notes|
|---|---|---|
|recommend.py|script called by the notebooks and |---|



## 3 Development environment
### 3.1 Environment setup

Note: this environment requires GPU. The cuda version of the local machine might have conflicts with tensorflow. This version has been tested with cuda 9.0.

```
conda create --name dev python=3.6.8
conda activate dev
pip install -r requirement_dev.txt
```

### 3.2 Insructions
|Notebook name                |Required data folder    |Time to run                   |
|---|---|---|
|- train.py: scripts to conduct grid search on model hyperparameters, the full experiments took >100 hrs on a machine with 2 gpus. |Currently the epoch is set to 2 and only 1 experiment will be conducted for demonstration purposes. To run the experiment,|
|1data_preprocessing_filtering|data_raw| 10 min|
|2data_preprocessing_clearning|data_raw|3 min|
|3data_preprocessing_dataset|data_raw|20 min|
|4features_engineering|data_raw|20 - 50 min, depending on gpu and internet speed, requires very powerful gpu|
|5features_topic|data_raw|2 min|
|6result_analysis|results|< 1 min|
| data_raw/|||



# Reference

https://github.com/huggingface/pytorch-transformers

