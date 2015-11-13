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



""":type: pd.DataFrame"""
# testData = pd.read_csv(fileNameTestData)
"""
Test model on unseen data. After each prediction step, you may update you model. This is not mandatory though.
"""

# Loading the trained model in case we do not want to train on new data
model = DataHandling.load_data(Constants.model_path + 'Model_NB.pickle')
os_list = DataHandling.load_data(Constants.model_path + 'os_list.pickle')
publisher_list = DataHandling.load_data(Constants.model_path + 'publisher_list.pickle')

#In case there is a new termVectors file we use this to extract features, I assume it is because articles will be updated.
feature_extraction = FeatureExtraction()
[article_word_count, word_tfidf, publishers, article_numbers] = feature_extraction.prepare_dictionary_article(termVectors)
# #For testing train data the same as test data
testData = train_data

article_popularity = DataHandling.load_data(Constants.model_path + 'article_popularity.pickle')
train_data = DataHandling.load_data(Constants.model_path + 'train_data_with_article_distances.pickle')

for (rowNum, row) in testData.iterrows():
    inputFeatures = row[["Publisher", "Osfamily", "ItemSrc", "UserID", "UserClicksAd"]]

    #Check if user has a history
    number_times_clicked = row["UserClicksAd"]

    #For users with no history pick the most popular item
    if number_times_clicked == 0:
        #Pick top 20 articles
        top_article_popularity = heapq.nlargest(20, enumerate(article_popularity), key=lambda x: x[1])
        popular_choice = random.choice(top_article_popularity)[0]

        #If publisher is the same as source pick again
        while str(publishers[popular_choice]) == str(int(inputFeatures['Publisher'])):
            top_article_popularity = heapq.nlargest(20, enumerate(article_popularity), key=lambda x: x[1])
            popular_choice = random.choice(top_article_popularity)[0]

        recommended_article_number = article_numbers[popular_choice]

    else:

        distance_df = article_word_count - article_word_count.loc[str(int(row['ItemSrc']))]
        distance_df = distance_df.abs().values.dot(word_tfidf)
        distance_array = np.reshape(distance_df, [len(distance_df), 1])

        best_match_probability = 0
        best_article_index = 0
        for i in xrange(len(distance_array)):
            if str(publishers[i]) != str(int(inputFeatures['Publisher'])):
                inputFeatures["ArticleDistance"] = distance_array[i]
                inputFeatures["Publisher"] = feature_extraction.get_publisher_feature(inputFeatures["Publisher"], publisher_list)
                inputFeatures["Osfamily"] = feature_extraction.get_os_feature(inputFeatures["Osfamily"], os_list)


                current_match_probability = model.series_predict_prob(inputFeatures)

                if current_match_probability[0][1] > best_match_probability:
                    best_match_probability = current_match_probability[0][1]
                    best_article_index = [i]

        recommended_article_number = article_numbers[i]

    print('Recommended article number is: ', recommended_article_number)


#     items = model.predict(inputFeatures)
#     model.trainOnline(row)
#     clicks += int(row["OsrItem"] in items[:topN])
#
# print("you got %d clicks with a CTR of %.5f"%(clicks, clicks/trainData.count(0)["Output"]))
#

