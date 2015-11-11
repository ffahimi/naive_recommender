import pandas as pd
import json
import os, sys
sys.path.append(os.getcwd()+"/..") # ensure python interpreter find files, may not be necessary

from MyModel import MYModel
from FeatureExtraction.FeatureExtraction import FeatureExtraction
from Utils import DataHandling
from Configuration.Constants import Constants

"""
Prepare data for trianing and testing
"""
fileNameTrainData = "data/train.csv"
fileNameTermVectors = "data/termVectorsPerPublisher.json"
# fileNameTestData = "data/test.csv" # well. obviously we dont hand this one out
topN = 5

termVectors = json.load(open(fileNameTermVectors))
trainData = pd.read_csv(fileNameTrainData)

#Feature Extraction part
#Running feature extraction to create word dictionary, word counts, article grouping (distances)
feature_extraction = FeatureExtraction()
feature_extraction.prepare_features(trainData, termVectors)

train_data = DataHandling.load_data(Constants.model_path + 'train_data_with_article_distances.pickle')

"""
prepare the model and test it
"""
model = MYModel()
model.trainBatch(train_data)


#Cross-fold validation of training model
scores = model.crossfoldValidation(train_data, 10)


#Prediction of Probabilities using Model (If needed to be used online, a threshold is needed)
predictions = model.predict(train_data)
DataHandling.store_data(Constants.model_path + 'predictions_prob.pickle', predictions)
train_data['PredictionProb'] = predictions
DataHandling.store_data(Constants.model_path + 'train_data_with_article_distances.pickle', train_data)

"""
#Pickle or CSV final training data (For testing and Matlab)
# train_data = DataHandling.load_data(Constants.model_path + 'train_data_with_prediction_prob.pickle')
# train_data.to_csv('prediction_file.csv')
"""



""":type: pd.DataFrame"""
# testData = pd.read_csv(fileNameTestData)
"""
Test model on unseen data. After each prediction step, you may update you model. This is not mandatory though.
"""
# for (rowNum, row) in testData.iterrows():
#     inputFeatures = row[["Publisher","Osfamily","ItemSrc","UserID","UserClicksAd"]]
#     items = model.predict(inputFeatures)
#     model.trainOnline(row)
#     clicks += int(row["OsrItem"] in items[:topN])
#
# print("you got %d clicks with a CTR of %.5f"%(clicks, clicks/trainData.count(0)["Output"]))

