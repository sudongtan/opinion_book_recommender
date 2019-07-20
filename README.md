# A Opinion-Diverse Book Recommender System
CS6460 Summer 2019
Xiaodong Tan (xtan74@gatech.edu)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sudongtan/cs6460_summer_2019/master)

# 1 Introduction
The project designs a recommender system that recommends books with diverse opinions to the users. By doing so, the users' confirmatin bias can be reduced and their critical thinking can be enhanced. The dataset used is the history and bibography books in the GoodRead Dataset. The user can give three inputs to the system.
 - user_id: int >=0, <= 45962
 - k (number of items to be recommended): int >=5 < 50
 - mode: string, 'random', 'interested_only', 'topic'
    - 'random': suitable for people who have higher tolerance of opinion diversity
    - 'interested only': suitable for people who have lower tolerance of opinion diversity
    - 'topic': the recommendations are based on the user's most interested topic

# 2 Recommendation Environment

## 2.1 Environment setup
### Method 1: 
It might take 1 - 3 minutes to launch it
https://mybinder.org/v2/gh/sudongtan/cs6460_summer_2019/master
### Method 2:
```
conda create --name rs python=3.6.8
conda activate rs
pip install -r requirements.txt
```
## 2.2 Instructions
|File name                |Insructions    |
|---|---|
|`final_project.ipynb`|Demostration of how to use the system to get recommendations. `data/` and `models` folders required.|
|`analysis_rating.ipynb`|Analysis of the rating models' performance. `results/` folder required. |
|`analysis_ranking.ipynb`|Analysis of the ranking results. `results/` folder required. Currently the sample size is set to be 10 for demostration purposes.|
|`app.py`|A simple user interface to get recommendations, run `python app.py` and open the link in the brower|

## 2.3 Source code
|File name                |Description|
|---|---|
|`recommend.py`|ranking algorithm and pipeline|
|`src/chainRec.py` |rating model constructor|
|`src/dataset.py` |rating model data spliting and sampling|


# 3 Development Environment
## 3.1 Environment setup

This environment requires GPU. The cuda version of the local machine might have conflicts with tensorflow. This version has been tested with cuda 10.

```
conda create --name dev python=3.6.8
conda activate dev
pip install -r requirement_dev.txt
```

### 3.2 Insructions
|File name                |Instruction    |Time to run                   |
|---|---|---|
|`data_raw`|[goodreads_book_authors.json](https://drive.google.com/uc?id=19cdwyXwfXx_HDIgxXaHzH0mrx8nMyLvC), [goodreads_book_series.json](https://drive.google.com/uc?id=1op8D4e5BaxU2JcPUgxM3ZqrodajryFBb), [goodreads_books_history_biography.json](https://drive.google.com/uc?id=1roQnVtWxVE1tbiXyabrotdZyUY7FA82W), [goodreads_interactions_history_biography.json ((1.6G))](https://drive.google.com/uc?id=10j181giCD94pcYynd6fy2U0RyAlL66YH) needed to be downloaded and put here|10 min - 20 min, depending on internet speed|
|`train.py`|scripts to conduct grid search on model hyperparameters,run `python train.py` | The full experiments took 100+ hours. Currently the epoch is set to 2 and only 1 experiment will be conducted for demonstration purposes. |
|`notebooks/1data_preprocessing_filtering.ipynb`|raw data required|10 min|
|`notebooks/2data_preprocessing_clearning.ipynb`|raw data required|3 min|
|`notebooks/3data_preprocessing_dataset.ipynb`|raw data required|20 min|
|`notebooks/4features_engineering.ipynb`|raw data required, powerful gpus required|20 - 50 min, depending on gpu and internet speed|
|`notebooks/4features_topics.ipynb`|raw data required|2 min|


# Reference
https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
https://github.com/MengtingWan/chainRec
https://github.com/awarebayes/RecNN
https://github.com/huggingface/pytorch-transformers
https://github.com/cemoody/lda2vec
