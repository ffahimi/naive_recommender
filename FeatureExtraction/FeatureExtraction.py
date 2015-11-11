__author__ = 'farshadfahimi'


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

    def prepare_training_article_distance(self, train_data, article_word_count, word_tdidf):

        for index, row in train_data.iterrows():
            distance_series = article_word_count.loc[str(int(row['ItemSrc']))] - article_word_count.loc[str(int(row['OsrItem']))]
            distance_series = distance_series.abs().values.dot(word_tdidf)

            train_data.loc[index, 'ArticleDistance'] = distance_series

        return train_data

    def prepare_features(self, train_data, term_vectors):

        #main runner for feature extraction
        print('... Preparing dataframe inclusing all words and tf-idf.')
        [article_word_count, word_cloud, word_tfidf] = self.prepare_dictionary_article(term_vectors)

        #calculate distance array of articles
        print('... Preparing training data inclusing the article distances between original article and recommended.')
        train_data = self.prepare_training_article_distance(train_data, article_word_count, word_tfidf)
        DataHandling.store_data(Constants.model_path + 'train_data_with_article_distances.pickle', train_data)

