__author__ = 'farshadfahimi'

import pandas as pd
import numpy as np
import json
import os, sys
sys.path.append(os.getcwd()+"/..") # ensure python interpreter find files, may not be necessary
import random
import heapq

from MyModel import MYModel
from FeatureExtraction.FeatureExtraction import FeatureExtraction
from Utils import DataHandling
from Configuration.Constants import Constants


"""
Prepare data for trianing and testing
# """
fileNameTrainData = "data/train.csv"
fileNameTermVectors = "data/termVectorsPerPublisher.json"
# fileNameTestData = "data/test.csv" # well. obviously we dont hand this one out
topN = 5

termVectors = json.load(open(fileNameTermVectors))
train_data = pd.read_csv(fileNameTrainData)

# Feature Extraction part
# Running feature extraction to create word dictionary, word counts, article grouping (distances)
feature_extraction = FeatureExtraction()
[train_data, article_popularity, os_list, publisher_list] = feature_extraction.prepare_features(train_data, termVectors)

#Taking only users that have historically clicked into training.
train_data = train_data.loc[train_data['UserClicksAd'] > 0]

#Balancing training data
#Since the dataset is imbalanced, we balance negative and positive samples for training.
# positive_samples = train_data.loc[train_data['Output'] == 1]
# positive_samples_count = train_data.loc[train_data['Output'] == 1].shape[0]
# negative_samples = train_data.loc[train_data['Output'] == 0]
# rows = random.sample(negative_samples.index, positive_samples_count)
# negative_samples = negative_samples.ix[rows]
# train_data = pd.concat([positive_samples, negative_samples])

"""
prepare the model and test it
"""
model = MYModel()
model.trainBatch(train_data)
DataHandling.store_data(Constants.model_path + 'Model_NB.pickle', model)


#Cross-fold validation of training model
scores = model.crossfoldValidation(train_data, 10)
print(scores)


"""
#Pickle or CSV final training data (For testing and Matlab)
# train_data = DataHandling.load_data(Constants.model_path + 'train_data_with_prediction_prob.pickle')
# train_data.to_csv('prediction_file.csv')
"""