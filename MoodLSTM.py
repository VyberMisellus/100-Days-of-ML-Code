# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:55:46 2018

@author: Isaac Csekey
"""

#LSTM to read in the pretrained word embeddings and learn patterns in text
import tensorflow as tf
import tflearn as tl
import gensim.models as w2v
import os
import re
from random import shuffle
import nltk
import codecs

os.chdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code")

#Loading a google word vectors library
svctrs= w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)
#Setting up training and testing data 

labelslist = subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate','mentalhealth','depression_help','depressionregimens','Anxiety']
path = r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code"
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
                    matrix_embed.append(svctrs[word])
                except KeyError:
                    pass
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
train = data[0:int(0.8*len(vectored))]
test = data[1+int(0.8*len(vectored)):len(vectored)]

trainX,trainY,testX,testY = []

for i in range(len(train)):
    temp = data[i]
    trainX.append(temp[0])
    trainY.append(temp[1])

for h in range(len(test)):
    temp = data[i]
    testX.append(temp[0])
    testY.append(temp[1])
#Padding the data for the different lengths of words
trainX = tl.data_utils.pad_sequences(trainX, maxlen = 1000, value = 0.)
testX = tl.data_utils.pad_sequences(testX, maxlen = 1000, value = 0.)

#Categorizing the labels
trainY = tl.data_utils.to_categorical(trainY)
testY = tl.data_utils.to_categorical(testY)

#BUILD THE NETWORK
net = tl.input_data([None, 400])
net = tl.embedding(net, input_dim = len(data), output_dim = 200)
net = tl.lstm(net, 200, dropout = 0.8)
net = tl.fully_connected(net, len(labelslist), activation = 'softmax')
net = tl.regression(net, optimizer = 'adam', learning_rate = 0.01, loss = 'categorical_crossentropy')

print("Model built")

#Training model 
model = tl.DNN(net, tensorboard_verbose = 0)
model.fit(trainX,trainY,validation_set = (testX,testY), show_metric = True, batch_size = 32)