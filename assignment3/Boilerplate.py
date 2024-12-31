
# Author: Swati Mishra
# Created: Sep 23, 2024
# License: MIT License
# Purpose: This python file includes boilerplate code for Assignment 3

# Usage: python support_vector_machine.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added boilerplate code

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate =learning_rate
        self.epoch = epoch
        self.C = C_value

        #initialize the weight matrix based on number of features 
        # bias and weights are merged together as one matrix
        # you should try random initialization
     
        self.weights = np.zeros(X.shape[1])

    def pre_process(self,):

        #using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input)

        Y_ = self.target 

        return X_,Y_ 
    
    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        loss=0

        # hinge loss implementation- start

        # Part 1

        # hinge loss implementatin - end
                
        return loss
    
    def stochastic_gradient_descent(self,X,Y):

        samples=0

        # execute the stochastic gradient des   cent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            #print epoch if it is equal to thousand - to minimize number of prints
            if epoch%1000 ==0:
                loss = self.compute_loss(features, output)
                print("Epoch is: {} and Loss is : {}".format(epoch, loss))

            #check for convergence -start
            
            # Part 1

            print("The minimum number of iterations taken are:",epoch)

            #check for convergence - end

            # below code will be required for Part 3
            
            # Part 3
            samples+=1

        print("Training ended...")
        print("weights are: {}".format(self.weights))

        # below code will be required for Part 3
        print("The minimum number of samples used are:",samples)

    def mini_batch_gradient_descent(self,X,Y,batch_size):

        # mini batch gradient decent implementation - start

        # Part 2

        # mini batch gradient decent implementation - end

        print("Training ended...")
        print("weights are: {}".format(self.weights))

    def sampling_strategy(self,X,Y):
        x = X[0]
        y = Y[0]
        #implementation of sampling strategy - start

        #Part 3

        #implementation of sampling strategy - start
        return x,y

    def predict(self,X_test,Y_test):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]
        
        #compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision - start
        # Part 2
        #compute precision - end

        #compute recall - start
        # Part 2
        #compute recall - end

def part_1(X_train,y_train):
    #model parameters - try different ones
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data

    # select samples for training

    # train model


    return my_svm

def part_2(X_train,y_train):
    #model parameters - try different ones
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data

    # select samples for training

    # train model


    return my_svm

def part_3(X_train,y_train):
    #model parameters - try different ones
    C = 0.001 
    learning_rate = 0.001 
    epoch = 5000
  
    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data

    # select samples for training

    # train model


    return my_svm

#Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv('data.csv')

# drop first and last column 
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()

#converting categorical variables to integers 
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

my_svm = part_1()

my_svm = part_2()

my_svm = part_3()

#normalize the test set separately
scalar = StandardScaler().fit(X_test)
X_Test_Norm = scalar.transform(X_test)

# testing the model
print("Testing model accuracy...")
my_svm.predict(X_Test_Norm,y_test)
