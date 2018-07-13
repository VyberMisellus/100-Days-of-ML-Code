# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:55:46 2018

@author: Isaac Csekey
"""

#LSTM to read in the pretrained word embeddings and learn patterns in text
import tensorflow as tf
import tflearn as tl
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import gensim.models as w2v
import os
import re
from random import shuffle
import nltk
import codecs
import numpy as np

os.chdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code")

#Loading a google word vectors library
svctrs= w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)
#Setting up training and testing data 

labelslist = subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','sad','hate']
path = r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods"
data = [] #For training and testing

#A function to get the data from the directories and clean it up a bit
def getData(subreddit_list):

    #Dictionary instance for ease of labelling
    labelled_data = dict()
    
    
    #Chunking through all the items in the subreddit list to open up the post files, clean them up, and add them to the 
    for item in subreddit_list:
        os.chdir(os.path.join(path,item))
        
        #An empty list to store all the posts in
        
        category_list = []
        
        
        #Going through all the files in the directory
        for root, dirs, files in os.walk("."):
            
            #Going through all the files MORE
            for filename in files:
                
                #An instance of a string that will act as a collector for all the text thrown into it
                collectionstring = str()
                
                #Embedding matrix for each post
                
                embed = []
                
                with codecs.open(filename, 'r', 'utf-8') as file:
                    try:
                        collectionstring += file.read()
                    except UnicodeDecodeError:
                        pass
                
                print("Opened ",filename)
                
                #Cleaning the data a little
                collectionstring = re.sub("[^a-zA-Z]"," ", collectionstring)
                
                collectionstring = collectionstring.lower()
                
                collectionstring = collectionstring.split()
                
                
                #Converting strings to their embedded vectors from the pre-trained word2vec model
                
                
                
                category_list.append(collectionstring)
                
        labelled_data[item] = category_list
        print(item, "loaded to dictionary\n")
        

    return labelled_data


#A function to take data from a dictionary and convert into a feature/label list of lists i.e. [[x,y],[x,y]...]
def toXY(dataDict):
    
    Labeled = []
    
    for key in dataDict.keys():
        templist = dataDict[str(key)]
        
        for post in templist:
            Labeled.append([post,str(key)])
        
    return Labeled

def vectorize(dataDictionary):
    
    total_XY = []
    
    for key in dataDictionary.keys():
        post_set = dataDictionary[key]
        
        for post in post_set:
            matrix_embed = []
            for word in post:
                try:
                    matrix_embed.append(svctrs[word].tolist)
                except KeyError:
                    pass
            length = 1000-len(matrix_embed)
            
            
            j = 0
            while j < length:    
                matrix_embed.append([0.0]*300) #padding the matrix
                j+=1
            #matrix_embed = np.array(matrix_embed).reshape(-1,300,1000,1)
            total_XY.append([matrix_embed, key])
            
    return total_XY
                
data_dict = getData(labelslist)

vectored = vectorize(data_dict)

#Randomizing the entries in the list 
   
data = vectored
shuffle(data)


#This is where the fun begins
#______________________________________________________________________________
#|                               LSTM TIME                                    | 
#|____________________________________________________________________________|

#Preparing the data for training and testing
train = data[0:int(0.9*len(data))]
test = data[int(0.9*len(data))+1:]

trainX=trainY=testX=testY = []
for item in train:
    trainX.append(item[0])
    trainY.append(item[1])
for item in test:
    testX.append(item[0])
    testY.append(item[1])


##BUILD THE NETWORK
network = input_data(shape=[None, 300, 1000, 1], name='input')
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
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

#
##Training model 
model = tl.DNN(network, tensorboard_verbose = 0)
print("Model built")
model.fit(trainX,trainY,validation_set = (testX,testY), show_metric = True, batch_size = 100)