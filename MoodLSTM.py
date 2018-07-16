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
#import nltk
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
    #All labels and features
    total_XY = []
    data_conv = dict()
    i = 0
    for item in labelslist:
        data_conv[item] = i
        i+=1
    
    #Through each key
    for key in dataDictionary.keys():
        
        #Each set of posts for each category
        post_set = dataDictionary[key]
        
        i = 0
        #Going through each post
        for post in post_set:
            
            #Making an embedding matrix from each string
            matrix_embed = []
            
            #Going through each word in the post
            for word in post:
                
                #Will only give embeddings for words in the embedding dictionary
                try:
                    matrix_embed.append(svctrs[word])
                except KeyError:
                    pass
            if len(matrix_embed) > 1000:
                matrix_embed = matrix_embed[:999]
            if len(matrix_embed)>0:
                #Defining the number of extra padded vectors for the embedding matrix
                length = 1000-len(matrix_embed)
                j = 0
                print("Before:",len(matrix_embed),"Internal:",type(matrix_embed[0]))
                #Padding the matrix
                while j < length:    
                    matrix_embed.append(np.array([0.0]*300)) 
                    j+=1
            
            
                matrix_embed = np.array(matrix_embed).reshape(-1,1000,300,1)
                print("After:",i,matrix_embed.shape)
                #matrix_embed = np.array(matrix_embed).reshape(-1,300,1000,1)
                total_XY.append([matrix_embed, data_conv[key]])
                i+=1
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

trainX=[]
trainY=[]
testX=[]
testY = []
for item in train:
    trainX.append(item[0])
    trainY.append(item[1])
for oitem in test:
    testX.append(oitem[0])
    testY.append(oitem[1])


#BUILD THE NETWORK
network = input_data(shape=[None, 1000, 300, 1], name='input')
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
model = tl.DNN(network, tensorboard_verbose = 0)
print("Model built")
model.fit(trainX,trainY,validation_set = (testX,testY), show_metric = True, batch_size = 100)