from random import shuffle
import numpy as np
import pandas as pd
from Classifier.NaiveBayes import NaiveBayes
from Utils import DataHandling
from Configuration.Constants import Constants

class MYModel:

    def __init__(self):
        self.classifier = NaiveBayes(1)

    def trainBatch(self, train_data):
        self.classifier.train(train_data)
        print('... Pickling the model')
        DataHandling.store_data(Constants.model_path + 'classifier_NB.pickle', self.classifier)

    def crossfoldValidation(self, train_data, number_of_folds):
        return self.classifier.classification_CV(train_data, number_of_folds)

    def trainOnline(self, row):
        """
        update the model on only one instance

        :param row: columns: "Output","OsrItem","Publisher","Osfamily","ItemSrc","UserID","UserClicksAd"
        :type row: pd.Series
        :return:
        """
        pass

    def predict(self, train_data, load_classifier=False):
        if load_classifier:
            self.classifier = DataHandling.load_data(Constants.model_path + 'classifier_NB.pickle')

        return self.classifier.classification(train_data)



        """
        select all item from term Vectors and shuffle them
        """
        # publisher = str(int(row["Publisher"]))
        # items = list(self.__termVectors[publisher].keys())
        # shuffle(items)
        # return np.array(items, dtype=float)