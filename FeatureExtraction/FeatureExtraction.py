__author__ = 'farshadfahimi'


import pandas as pd
import numpy as np
from Utils import DataHandling
from Configuration.Constants import Constants

class FeatureExtraction:
    def __init__(self):
        pass

    def prepare_dictionary_article(self, term_vectors):

        #create a list of words from available articles
        word_cloud = []
        word_tfidf = []
        publishers = []
        article_numbers = []

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
                publishers.append(publisher)
                article_numbers.append(article_number)
                for word in words:
                    article_word_count.loc[article_number, word] = 1

        #removing nan values
        article_word_count = article_word_count.fillna(0)

        return article_word_count, word_tfidf_series, publishers, article_numbers

    def prepare_training_article_distance(self, train_data, article_word_count, word_tdidf):

        for index, row in train_data.iterrows():
            distance_series = article_word_count.loc[str(int(row['ItemSrc']))] - article_word_count.loc[str(int(row['OsrItem']))]
            distance_series = distance_series.abs().values.dot(word_tdidf)

            train_data.loc[index, 'ArticleDistance'] = distance_series

        return train_data

    def article_popularity(self, train_data, article_word_count, article_numbers):

        article_popularity = []
        for i in xrange(len(article_numbers)):
            # Create a feature for article popularity (thresholds are guesses at the moment)
            #How many times in past this article is read
            article_read_number = (train_data.loc[train_data['ItemSrc'] == int(article_numbers[i])]).shape[0]
            article_popularity.append(article_read_number)

        return article_popularity

    def normalize_features(self, train_data):

        os_list = []
        publisher_list = []

        #unique operating system categories
        unique_oss = train_data['Osfamily'].unique()
        for i in xrange(len(unique_oss)):
            os_list.append(unique_oss[i])
            train_data.ix[train_data.Osfamily == unique_oss[i], 'Osfamily'] = i

        #unique publisher categories
        unique_publisher = train_data['Publisher'].unique()
        for i in xrange(len(unique_publisher)):
            publisher_list.append(unique_publisher[i])
            train_data.ix[train_data.Publisher == unique_publisher[i], 'Publisher'] = i
            print train_data.loc[train_data['Publisher'] == i]

        return train_data, os_list, publisher_list

    def get_os_feature(self, os_number, os_list):
        for i in xrange(len(os_list)):
            if str(int(os_number)) == str(os_list[i]):

                return str(i)

        return str(0)

    def get_publisher_feature(self, publisher_number, publisher_list):
        for i in xrange(len(publisher_list)):
            if str(int(publisher_number)) == str(publisher_list[i]):

                return str(i)

        return str(0)

    def prepare_features(self, train_data, term_vectors):

        #main runner for feature extraction
        print('... Preparing dataframe inclusing all words and tf-idf.')
        [article_word_count, word_tfidf, publishers, article_numbers] = self.prepare_dictionary_article(term_vectors)

        #Normalize Osfamily and Publisher
        [train_data, os_list, publisher_list] = self.normalize_features(train_data)
        DataHandling.store_data(Constants.model_path + 'os_list.pickle', os_list)
        DataHandling.store_data(Constants.model_path + 'publisher_list.pickle', publisher_list)

        #Calculate article popularity
        article_popularity = self.article_popularity(train_data, article_word_count, article_numbers)
        DataHandling.store_data(Constants.model_path + 'article_popularity.pickle', article_popularity)

        #calculate distance array of articles
        print('... Preparing training data including the article distances between original article and recommended.')
        train_data = self.prepare_training_article_distance(train_data, article_word_count, word_tfidf)
        DataHandling.store_data(Constants.model_path + 'train_data_with_article_distances.pickle', train_data)

        return train_data, article_popularity, os_list, publisher_list
