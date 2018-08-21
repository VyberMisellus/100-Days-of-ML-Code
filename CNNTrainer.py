# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:34:15 2018

@author: isaac
"""
import tflearn as tl
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import os
import pickle
from random import shuffle, seed
#Grabbing Data
os.chdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods")

print("Loading numpy data")
x = np.load("vectored_data.npy")

print("Loading labeled data")
with open("labels.txt",'rb') as fp:
    y = pickle.load(fp)
#y = np.array(y)
#Splitting into training and testing data
print("Shuffling data...")
seed(42)
shuffle(x)
shuffle(y)

print("Splitting data")
trainX = x[0:int(0.9*2643)]
trainY = y[0:int(0.9*2643)]
testX = x[int(0.9*2643):]
testY = y[int(0.9*2643):]
#CNN

print("Building network...")
#BUILD THE NETWORK
network = input_data(shape=[None, 300, 300, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

####Training model 
model = tl.DNN(network)
print("Model built")
model.fit(trainX,trainY,validation_set = (testX,testY), show_metric = True, batch_size = 50)