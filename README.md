# A preference-inconsistent book recommender
CS6460 Summer 2019
Xiaodong Tan (xtan74@gatech.edu)

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
|File name                |Insructions    |Notes                   |
|---|---|---|
|final_project.ipynb|demostrate how to use the system to get recommendations|---|
|analysis_rating.ipynb|analysis of the rating models' performance, requires `results/` folder|---|
|analysis_ranking.ipynb|analysis of the ranking results. |---|
|app.py|a simple user interface to get recommendations, run `python app.py` and open the link in the brower|---|

## 2.3 Source code
|File name                |Required data folder    |
|---|---|
|recommend.py|ranking algorithm and pipeline|
|src/chainRec.py |rating model constructor|
|src/chainRec.py |rating model data spliting and sampling|


# 3 Development
## 3.1 Environment setup

This environment requires GPU. The cuda version of the local machine might have conflicts with tensorflow. This version has been tested with cuda 9.0.

```
conda create --name dev python=3.6.8
conda activate dev
pip install -r requirement_dev.txt
```

### 3.2 Insructions
|File name                |Instruction    |Time to run                   |
|---|---|---|
|data_raw|folder to put the raw dataset|---|
|train.py|scripts to conduct grid search on model hyperparameters,run `python train.py` | The full experiments took 100+ hours. Currently the epoch is set to 2 and only 1 experiment will be conducted for demonstration purposes. |
|notebooks/1data_preprocessing_filtering.ipynb|raw data required|10 min|
|notebooks/2data_preprocessing_clearning.ipynb|raw data required|3 min|
|notebooks/3data_preprocessing_dataset.ipynb|raw data required|20 min|
|notebooks/4features_engineering.ipynb|raw data required, powerful gpus required|20 - 50 min, depending on gpu and internet speed|
|notebooks/4features_topics.ipynb|raw data required|2 min|


# Reference

https://github.com/huggingface/pytorch-transformers

