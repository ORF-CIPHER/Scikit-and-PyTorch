import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import copy

# sub classfier class
class sub_logReg:
    def __init__(self, etta, n_iter):
        self.etta = etta
        self.n_iter = n_iter
        self.iter_error = np.int64(0)
        self.total_error = np.zeros((n_iter,), dtype=int)
        self.objective_function = np.float32(0)
        self.activation_function = np.float32(0)
        
    def fit(self, x, y):
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.bias = np.float64(0.0)

        for i in range(self.n_iter):
            for feature,target in zip(x,y):
                prediction = self.predict_output(feature)
                self.optimizer(prediction, target, feature)
            self.total_error[i] = self.iter_error
            self.iter_error = np.int64(0)

    def predict_output(self, feature):
        self.objective_function = self.weights.dot(feature) + self.bias
        self.activation_function = 1/(1 + np.exp(-1*self.objective_function))
        if self.activation_function >= 0.5:
            return 1
        else :
            return 0

    def batch_predict(self, features):
        test_results = np.empty(features.shape[0], dtype = np.float64)
        for idx,feature in enumerate(features):
            test_results[idx] = self.predict_output(feature)
        return test_results

    def optimizer(self, prediction, target, feature):
        if prediction!= target:
            self.iter_error += 1
        error = target - self.activation_function
        for j in range(self.weights.shape[0]):   
            self.weights[j] += self.etta*error*feature[j] 
        self.bias += self.etta*error


#main classfier
class LogisticRegression:
    def __init__(self, etta, n_iter):
        self.etta = etta
        self.n_iter = n_iter
        self.subClassifiersList = []
        self.error = np.int64(0)
        self.number_of_subClassfiers_required = 0
        self.decision_array = []
        self.keys = {}
        self.originalDataset = None

    def fit(self, x, y):
        self.number_of_subClassfiers_required = len(np.unique(y))
        self.decision_array = np.zeros(len(np.unique(y)), dtype=float)

        for i in range(self.number_of_subClassfiers_required):
            self.subClassifiersList.append(sub_logReg(self.etta, self.n_iter))
            self.keys[i] = np.unique(y)[i]

        standardized_x = self.standardize_dataset(x)
        self.originalDataset = copy.deepcopy(x)
        

        for i in range(self.number_of_subClassfiers_required):
            specific_target_dataset = self.create_subClassfier_training_dataset(y, np.unique(y)[i])
            standardized_x, specific_target_dataset = self.shuffle_dataset(standardized_x, specific_target_dataset)
            self.subClassifiersList[i].fit(standardized_x, specific_target_dataset)
            standardized_x = self.standardize_dataset(x) # to get the data in original order

        print("Training carried out successfully")

        
    def create_subClassfier_training_dataset(self, y, key):
        target = np.where(y==key, 1, 0)
        return target

    def standardize_dataset(self, x):
        standardized_x = copy.deepcopy(x);
        for column in range(x.shape[1]):
            standardized_x[:, column] = ((x[:, column]) - np.mean(x, axis = 0)[column] ) / (np.std(x, axis = 0)[column])
        return standardized_x

    def shuffle_dataset(self, x, y):
        if(x.shape[0] == y.shape[0]):
            y = y.reshape((y.shape[0], 1))
            input_and_output = np.hstack((x,y))
            np.random.shuffle(input_and_output)
            return input_and_output[:, :-1], input_and_output[:, -1]
        else:
            print("Array size mismatch")


    def predict_output(self, feature):
        #this function should only be used by other functions, and should not be invoked externally

        for i in range(self.number_of_subClassfiers_required):
            self.decision_array[i] = self.subClassifiersList[i].predict_output(feature)
        
        return (self.decision_array.argmax() + 1) # because class starts from 1, not 0

    # only this function should be used for testing, even if the test data has only 1 sample
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
        test_results = np.empty(features.shape[0])
        
        for idx,feature in enumerate(features):
            test_results[idx] = self.predict_output(feature)
        return test_results

    def calculate_accuracy(self, target, prediction):
        if target.shape[0] != prediction.shape[0]:
            print("array size mismatch")
            return
        for idx in range(0, target.shape[0]):
            if(target[idx] != prediction[idx]):
                self.error += 1
        print("Accuracy of the model is = {}%".format(((target.shape[0] - self.error)/target.shape[0])*100))

    def plot_training_session(self):
        plt.rcParams['figure.figsize'] = [18, 6]
        for i in range(self.number_of_subClassfiers_required):
            plt.subplot(1,3,i+1)
            plt.title("Classfier for - {}".format(self.keys[i]))
            plt.xlabel('Epochs')
            plt.ylabel('Errors per iteration')
            plt.plot(range(1, len(self.subClassifiersList[i].total_error) + 1), self.subClassifiersList[i].total_error, marker='o')

