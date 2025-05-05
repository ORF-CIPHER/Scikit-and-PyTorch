import numpy as np 
import pandas as pd 
import copy
import math


class KNN:
    def __init__(self, k):
        self.k = k
        self.nearest_k_index = np.zeros((k,), dtype=int)
        self.decision_array = np.empty(k, dtype=object)
        self.nearest_k_distances = np.zeros((k,), dtype=float)
        self.originalDataset = None # stores the unstandardized features, no target column
        self.standardizedDataset = None # stores the standardized features along with target column, used for finding k nearest neighbours
        self.error = np.int64(0)

    def fit(self, x, y):
        standardized_x = self.standardize_dataset(x)
        y = y.reshape(y.shape[0],1)
        self.standardizedDataset = np.hstack((standardized_x, y))
        self.originalDataset = copy.deepcopy(x)


    def standardize_dataset(self, x):
        standardized_x = copy.deepcopy(x);
        for column in range(x.shape[1]):
            standardized_x[:, column] = ((x[:, column]) - np.mean(x, axis = 0)[column] ) / (np.std(x, axis = 0)[column])
        return standardized_x

    def calculateEuclideanDistance(self, p1, p2):
        distance = 0
        if(p1.shape[1] != p2.shape[1]):
            print("Array shape mismatch from calc distance")
            return
    
        for i in range(p1.shape[1]):
            distance += math.pow((p1[0, i] - p2[0, i]), 2)
        distance = math.sqrt(distance)
        return distance

    def predict_output(self, feature):
        distance_dataset = np.zeros((self.standardizedDataset.shape[0],), dtype=float) 

        for i in range(self.standardizedDataset.shape[0]):
            distance_dataset[i] = self.calculateEuclideanDistance(feature, self.standardizedDataset[i, : -1].reshape(1, feature.shape[1]))

        self.nearest_k_index = distance_dataset.argsort()[:self.k] # returns a array consisting of indexes of values arranged in ascending order
        for i in range(self.k):
            self.decision_array[i] = self.standardizedDataset[self.nearest_k_index[i], -1]

        # below code is used to find the majority vote
        pds = pd.Series(self.decision_array)
        counts = pds.value_counts()
        return counts.index[0]

        

    def batch_predict(self, features):
        if(len(features.shape) != 2):
            print("Array shape mismatch")
            return
        
        if(features.shape[1] != self.originalDataset.shape[1]):
            print("\nIncomplete data, please ensure that all features that were given while training are given while testing\n")
            return

        #standardizing the test data
        for column in range(features.shape[1]):
            features[:, column] = ((features[:, column]) - np.mean(self.originalDataset, axis = 0)[column] ) / (np.std(self.originalDataset, axis = 0)[column])
        test_results = np.empty(features.shape[0], dtype=object)
        
        for idx,feature in enumerate(features):
            test_results[idx] = self.predict_output(feature.reshape(1, self.originalDataset.shape[1]))
        return test_results

    def calculate_accuracy(self, target, prediction):
        if target.shape[0] != prediction.shape[0]:
            print("array size mismatch")
            return
        for idx in range(0, target.shape[0]):
            if(target[idx] != prediction[idx]):
                self.error += 1
        print("Accuracy of the model is = {}%".format(((target.shape[0] - self.error)/target.shape[0])*100))

