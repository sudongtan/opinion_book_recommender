import os
import sys
sys.path.append('./src/')
from dataset import Dataset
from chainRec import chainRec
import argparse


def train_model(DATA_NAME, n_stage, method, embed_size, lbda, lr):
    myData = Dataset(DATA_NAME, n_stage)
    myData.split_train_test(seed=1234, max_validation_test_samples=100000)

    validation_samples = myData.sampling_validation()
    training_samples = myData.sampling_training(method="edgeOpt_stage")
        
    myModel = chainRec(myData.n_user, myData.n_item, myData.n_stage, myData.DATA_NAME)
    myModel.config_global(MODEL_CLASS="edgeOpt_stage", HIDDEN_DIM=embed_size, 
                            LAMBDA=lbda, LEARNING_RATE=lr, BATCH_SIZE=256,
                            target_stage_id=(n_stage-1))
    myModel.load_samples(training_samples, validation_samples)
    myModel.train_edgeOpt()
    
    myModel.evaluation(myData.data_test, myData.user_item_map, topK=10)       
        
    
if __name__ == '__main__':

    nStage = 3
    dataset = 'book-3stage'
    method = 'chainRec_stage'
    lrs = [0.001]#[0.001, 0.0005, 0.0001]
    l2s = [0.2] #[0.2, 0.1, 0.15, 0.05, 0.02, 0.01]
    embedSizes = [16] #16, 32, 64, 128]
    for embedSize in embedSizes:
        for l2 in l2s:
            for lr in lrs:
                train_model(dataset, nStage, method, embedSize, l2, lr)
    
    