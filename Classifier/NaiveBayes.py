__author__ = 'farshadfahimi'
from datetime import datetime

from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class NaiveBayes:
    def __init__(self, alpha):
        self.classifier = MultinomialNB(alpha=alpha, fit_prior=False)
        self.creation_ts = datetime.now()

    def train(self, training_set):

        features = []
        classes = []
        for index, entry in training_set.iterrows():
            features.append(entry[['Osfamily', 'Publisher', 'ArticleDistance']])
            classes.append(entry['Output'])

        print('... Training the model')
        self.classifier.fit(features, classes)

    def classification(self, train_data):

        features = []

        for index, entry in train_data.iterrows():
            features.append(np.array(entry[['Osfamily', 'Publisher', 'ArticleDistance']]))

        print('... Started classification')
        results_prob = self.classifier.predict_proba(features)
        return results_prob

    def classification_CV(self, train_data, number_of_folds):

        features = []
        for index, entry in train_data.iterrows():
            features.append(entry[['Osfamily', 'Publisher', 'ArticleDistance']])

        print('... Started crossfold validation')
        score = cross_validation.cross_val_score(self.classifier, features, train_data.Output, cv=number_of_folds,
                                                 scoring='f1')
        return score

    def set_priors(self, priors):
        # sets priors of naive bayes component to given values
        if priors[0] + priors[1] == 1:
            self.classifier.set_params(class_prior=priors)
        else:
            print 'Priors should some up to 1.'
            exit()
