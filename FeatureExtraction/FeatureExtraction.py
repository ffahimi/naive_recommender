__author__ = 'farshadfahimi'

import numpy as np
import pandas as pd
from Utils import DataHandling
from Configuration.Constants import Constants

class FeatureExtraction:
    def __init__(self):
        pass

    def prepare_dictionary_article(self, term_vectors):

        #create a list of words from available articles
        word_cloud = []
        word_tfidf = []

        for publisher, article in term_vectors.iteritems():
            for article_number, words in article.iteritems():
                for word in words:
                    if word not in word_cloud:
                        word_tfidf.append(words[word]['tf-idf'])
                        word_cloud.append(word)

        #create dataframe to keep wordcount per article
        article_word_count = pd.DataFrame(columns=word_cloud)
        word_tfidf_series = pd.Series(word_tfidf)

        for publisher, article in term_vectors.iteritems():
            for article_number, words in article.iteritems():
                for word in words:
                    article_word_count.loc[article_number, word] = 1

        #removing nan values
        article_word_count = article_word_count.fillna(0)

        return article_word_count, word_cloud, word_tfidf_series

    def prepare_user_word_count(self, train_data, article_word_count, word_cloud):


        #iterate through unique users to create wordcount
        unique_user_ids = pd.unique(train_data.UserID.ravel())

        #Create a dataframe to keep user wordcount
        user_word_count_dataframe = pd.DataFrame(index=unique_user_ids, columns=word_cloud)

        count = 0
        for user_id in unique_user_ids:
            user_read_articles = []
            user_id_dataframe = train_data[train_data['UserID'] == user_id]

            for index, row in user_id_dataframe.iterrows():

                if row['ItemSrc'] not in user_read_articles:
                    user_read_articles.append(str(int(row['ItemSrc'])))

            user_word_count = article_word_count.loc[user_read_articles].sum(axis=0)
            user_word_count[user_word_count > 0] = 1

            print_full(user_word_count)
            count += 1
            print count
            print(user_id)
            user_word_count_dataframe.append(user_word_count.T, ignore_index=True)
            user_word_count_dataframe[[user_id, user_word_count.loc == 1]] = 1

            # print_full(user_word_count_dataframe.loc[[user_id]])

            # # user_word_count_dataframe.loc[user_id, user_word_count.loc == 1] = 1
            # print_full(user_word_count_dataframe)

            # for article_number in user_read_articles:
            #     user_word_count_dataframe.loc[user_id, article_word_count.loc[str(article_number)] == 1] = 1

        user_word_count_dataframe.fillna(0)
        DataHandling.store_data(Constants.model_path + 'user_word_count.pickle', user_word_count_dataframe)

    def prepare_training_article_distance(self, train_data, article_word_count, word_tdidf):

        for index, row in train_data.iterrows():
            distance_series = article_word_count.loc[str(int(row['ItemSrc']))] - article_word_count.loc[str(int(row['OsrItem']))]
            distance_series = distance_series.abs().values.dot(word_tdidf)

            train_data.loc[index, 'ArticleDistance'] = distance_series

            # print train_data[1:index].UserID == row[6]
            # iloc[train_data[train_data['UserID'] == row[6]]]

            # print train_data.loc[index, 'UserClicksAd']

        return train_data

    def prepare_features(self, train_data, term_vectors):

        #main runner for feature extraction
        [article_word_count, word_cloud, word_tfidf] = self.prepare_dictionary_article(term_vectors)


        # self.prepare_user_word_count(train_data, article_word_count, word_cloud)

        #calculate distance array of articles
        # train_data = self.prepare_training_article_distance(train_data, article_word_count, word_tfidf)
        # DataHandling.store_data(Constants.model_path + 'train_data_with_article_distances.pickle', train_data)


# def print_full(x):
#         pd.set_option('display.max_rows', len(x))
#         print(x)
#         pd.reset_option('display.max_rows')