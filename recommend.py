import os
import sys
sys.path.append('./src/')
from chainRec import chainRec
import argparse
import tensorflow as tf
import numpy as np
from itertools import permutations
import json
from collections import Counter

MODEL_DIR = "./models/"

book_idx_id = []
user_idx_id = []

book_idx_titles = json.load(open('data/book_idx_title.json', 'r'))
user_idx_book_idx = json.load(open('data/user_idx_book_idx.json', 'r'))
book_idx_topics = json.load(open('data/book_idx_topics.json', 'r'))
topic_book_idx = json.load(open('data/topic_book_idx.json', 'r'))

def load_model():
    model_name = "book-3stage.edgeOpt_stage.dim.128.lambda.0.01.lr.0.0001"
    lbda = 0.01
    lr = 0.0001
    embed_size = 128
    n_stage = 3
    n_user = 45963
    n_item = 13521
    DATA_NAME = 'book-3stage'

    myModel = chainRec(n_user, n_item, n_stage, DATA_NAME)
    myModel.config_global(MODEL_CLASS="edgeOpt_stage", HIDDEN_DIM=embed_size, 
                            LAMBDA=lbda, LEARNING_RATE=lr, BATCH_SIZE=256,
                            target_stage_id=(n_stage-1))

    session = tf.Session()
    u, i, j, li, lj, s, logloss, optimizer, valiloss = myModel.model_constructor(n_user, n_item, n_stage, embed_size, lbda, lr)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MODEL_DIR + model_name + ".model.ckpt")
    return session, s, u

def predict(model, user_id):
    session, s, u = model
    scores = np.array(s.eval(session=session, feed_dict={u:[user_id]}))
    return scores


def calculate_diversity(scores):
    n = len(scores)
    distances = 0
    for s in scores[:-1]:
        for j in scores[1:]:
            distance = abs(s - j)
            distances += distance

    return distances/n/(n-1)/2

def rank(user_id, scores, k=10, mode='random', diversity_preference=0, seed=1000):
    np.random.seed(seed)
    interest_scores = scores[:, :, 1].reshape(-1)
    opinion_scores = scores[:, :, 2].reshape(-1)
    interests = np.copy(interest_scores)
    interests[interests >= 1] = 1
    interests[interests < 1] = 0
    opinions = np.copy(opinion_scores)
    opinions[opinions >= 1] = 1
    opinions[opinions < 1] = 0

    s_interest_scores = []
    for s in interest_scores:
        if s >= 1:
            new_s = (s - 1)/(interest_scores.max() - 1)*0.5 + 0.5
        else:
            new_s = (s - 1)/(1  - interest_scores.min())*0.5
        s_interest_scores.append(new_s)
    s_interest_scores = np.array(s_interest_scores)
    s_opinion_scores = (opinion_scores - opinion_scores.min())/(opinion_scores.max() - opinion_scores.min())

    negative_candidates = np.where((interests==1) & (opinions==0))[0]
    positive_candidates = np.where((interests==1) & (opinions==1))[0]
    all_candidates = list(range(13521))
    interested_caididates = np.where(interests == 1)[0]

    if mode == 'interested_only':
        candidates = interested_caididates

    elif mode == 'random':
        candidates = all_candidates

    elif mode == 'topic':
        read = user_idx_book_idx[str(user_id)]
        topics = []
        for book_idx in read:
            topic = book_idx_topics[str(book_idx)]
            topics += topic
        if len(topics) == 0:
            chosen_topic = np.random.randint(0, 15)
        else:
            chosen_topic = Counter(topics).most_common(1)[0][0]

        candidates = topic_book_idx[str(chosen_topic)]  
        candidates = list(set(candidates) & set(interested_caididates))

    #print('here', candidates)
    sub_lists = [np.random.choice(candidates, k, replace=False) for _ in range(200)]
    max_metric = 0
    best_diversity = 0
    best_relevance = 0
    for sub_list in sub_lists:
        titles = [book_idx_titles[str(i)] for i in sub_list]
        fictional = False
        for word in ['novel', "mysteries"]:
            for title in titles:
                if word in title.lower():
                    fictional = True
        if fictional:
            continue

        relevance = np.average(s_interest_scores[sub_list])
        diversity = calculate_diversity(opinions[sub_list])

        top_diversity = calculate_diversity([1]*int(k/2) + [0]*(k-int(k/2)))
        diversity /= top_diversity
  
        diversity_weight = 0.5
        metric = diversity_weight * diversity + (1 - diversity_weight) * relevance

        if metric >= max_metric:
            recommended_list = sub_list
            best_diversity = diversity
            best_relevance = relevance
            max_metric = metric

    return best_relevance, max_metric, recommended_list


def recommend(model, user_id, k=10, mode='random', seed=1000, display=True):
    try:
        user_id = int(user_id)
        k = int(k)
    except TypeError:
        raise ValueError(f"Number of books to be recommended 'k' should be an integer, got {type(k)}")

    try:
        user_id = int(user_id)
    except TypeError:
        raise ValueError(f"The user id should be an integer, got {user_id}")

    if k < 5 or k > 50:
        raise ValueError(f"Number of books to be recommended 'k' should be in [5, 50], got {k}")

    if user_id >= 45963:
        raise ValueError(f"The user id should be in [0, 45962], got {user_id}")

    if not mode in ['random', 'interested_only', 'topic'] :
        raise ValueError(f"The mode should be 'random', 'interested_only' or 'topic', got '{mode}'")

    scores = predict(model, user_id)
    interest_scores = scores[:, :, 1].reshape(-1)
    opinion_scores = scores[:, :, 2].reshape(-1)
    interests = np.copy(interest_scores)
    interests[interests >= 1] = 1
    interests[interests < 1] = 0
    opinions = np.copy(opinion_scores)
    opinions[opinions >= 1] = 1
    opinions[opinions < 1] = 0

    relevance, metric, recommended_list = rank(user_id, scores, k, mode, seed)
    nliked = recommended_list[opinion_scores[recommended_list] < 1]
    liked = recommended_list[opinion_scores[recommended_list] >= 1]
    interested = recommended_list[interest_scores[recommended_list] > 1]

    liked_books = [book_idx_titles[str(i)] for i in liked]
    nliked_books = [book_idx_titles[str(i)] for i in nliked]
    nliked_books = [book_idx_titles[str(i)] for i in nliked]

    if display:
        if mode == 'random':
            print(f'The system selects candidate books to that you might be interested in')
        elif mode == 'interested_only':
            print(f'The system selects candidate books to that you are very likely to be interested in')
        elif mode == 'topic':
            print(f'The system selects candidate books based on your most interested topic')

        print(f'The system recommends {k} books to you')
        print()
        print('==========')
        print('Books you might agree with')
        print()
        for book in liked_books:
            print(book)
        print('==========')
        print('Books you might disagree with')
        print()
        for book in nliked_books:
            print(book)

    return liked_books, nliked_books, len(interested), metric, relevance



