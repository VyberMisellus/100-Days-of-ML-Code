# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:09:42 2018

@author: isaac
"""

#This is a demonstration of the power of latent vector representation in Natural Language Processing
#There is a deep neural network tasked with classifying a dataset of 
#text, extracted from reddit posts. Each subreddit is different, and the text
#classifiers are required to classify which subreddit a post or snippet of text
#belongs to. Applications of semantic text prediction include more refined sentiment analysis,
#as well as help watch for potentially dangerous individuals based on their posts
#online, as some more controversial subreddits such as Braincels have been included
#in the dataset.

import os #for file extraction
import numpy as np #for any useful math
from keras.utils import to_categorical #to one-hot encode the labels
from gensim.models import Word2Vec as w2v #the latent vector encoder, able to convert text to 300-dim. vector representations
import tensorflow as tf #for gpu acceleration

#Importing stop words and tokenizing function for sentence deconstruction
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

#importing keras layers for neural networks
from keras.models import Sequential, load_model, Model
from keras.layers import Dense,Conv1D, GlobalAveragePooling1D, MaxPooling1D, Input, concatenate


def predictor(string, word_length, punx, w2vmodel, stop_words, model,keyslist):
    
    #This function is used for user evaluation of the models, and takes raw text input,
    #encodes any words in the vocabulary, and predicts which subreddit the text is most like
    string=string.lower() #convertin all to lower case for cleanliness
    tempsent = word_tokenize(string) #separating words
    blank = [] #the list of word vectors
    for w in tempsent:
               
        try:
            blank.append(w2vmodel[w]) #sometimes not all words are in word2vec vocabulary
        except:
            pass

    final = easypad(blank,word_length) #a quick truncation of any words past what the model was trained to handle
    
    final=np.array(final) #converting to array for the proper data type 
    try:
        final=final.reshape([1,300,300]) #more datatype conversions 
        prediction = model.predict(final)
        answer = np.argmax(prediction) #get the answer with highest probability
    
        for k,v in keyslist.items():
            if answer == v:
                print(k,' is the prediction')
    except ValueError:
        print("Type in something else.")
        
        
def padder(vector, word_length):
    #Padding function to properly format training data
    returned = []
    while True: 
        if len(vector)>=word_length:
           returned.append(vector[:word_length])
           vector = vector[word_length:] #Split vector matrices up into chunks of
                                         #the right dimension
        else:
            while len(vector)<word_length: #Pad the rest of the 300x300 matrix 
                                           #with zeros so that the model can 
                                           #still accept the data
                vector.append([0]*300)
            break
    return returned

def easypad(matrix, length): #A quick padding tool that can accept up to 300 words
    if len(matrix) >= length:
        matrix=matrix[:300]
    else:
        while len(matrix)<length:
            matrix.append(np.array(300*[0]))
    return matrix

#Loading models and data:
    #The Google word2vec library
print("Loading word2vec model...")
w2vmodel = w2v.load('C:/Users/isaac/Documents/GitHub/100-Days-of-ML-Code/reddit_model_v2.model')
    #The NLTK bag of stop words
stop_words = set(stopwords.words('english')) 
    #The main working direcory
mother = r'C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods'
    #Puncuation bag for cleaning, punx is having problems downloading 
punx = ['!','#','$','%','&','(','.',')','*','+','-','.','/:',';','<','=','>','?','@','[','\]','^','_','`','{','|','}','~',')']

#Move to the directory
os.chdir(mother)

#Initialize a master-legend of features and labels
legend = dict()
#Creating a seperate list for just features
features =[]

#Walking through the entire directory
for root, dirs, files in os.walk("."):
        os.chdir(mother)
        #Naming the label after the directory
        legend[root] = []
        print("Reading: ",root)
        #Going through all the files
        i=0
        j=0
        k=0
        for filename in files:
            #print(root)
            #Use current location
            spot = os.path.join(mother,root)
            with open(os.path.join(spot,filename),'r') as file:
                #Reading in the file
                a = file.read()
            a=a.lower()
            #Removing punctuation 
            for item in punx:
                if item in a:
                    a = a.replace(item,' ')
                    
            
            #Tokenizing the temporary sentence
            tempsent = word_tokenize(a)
            #The temporary list of all relevant word vectors
            blank = []
            #For debugging and estimating how 'good' the w2v model is
            example = []
            #Encoding each word: a crash course
            for w in tempsent:
                
                try:
                    blank.append(w2vmodel[w])
                    example.append(w)
                except KeyError:
                    j+=1
                    pass
                finally:
                    pass
            #print(example)
            #Get rid of noisy empty lists
            if blank == []:
                k +=1
            else:
                #If all good, pad the vectors
                good = padder(blank,300)
                for entry in good:
                    legend[root].append(entry)
                    features.append(entry)
                    i+=1
        print(i, " good files found.")
        print(j, ' unexpected errors found.')
        print(k, ' dead files found.')
#Making a dictionary to translate to index
i=0
keyslist = dict()
for key in legend.keys():
    keyslist[key] = i
    i+=1

#Using the to-index dictionary to one-hot the labels
initial = []        
for k,v in legend.items():
    for entry in v:
        initial.append(keyslist[k])
labels = to_categorical(initial)

features = np.array(features)
#Randomize features and labels
idx = np.random.permutation(len(features))
x,y = features[idx], labels[idx]


#The first neural network being used is a convolutional neural network, which 
#is usually the type used for image classification, however the semantic vector representations 
#of the words allow matrix convolution to happen, as well as simplify the problem by
#providing consistent latent vector representations of the data. However, since the latent
#representations themselves cannot be fully realized by a human, an inception model is used
#to add robustness and scaling differences to the layers of the neural net. 
with tf.device('/gpu:0'):
    ins = Input(shape = (300,300))

    tower1 = Conv1D(64,1,activation='relu',padding = 'same')(ins)    #
    tower1 = Conv1D(64,3,activation='relu',padding = 'same')(tower1) #
    tower1 = Conv1D(64,3,activation='relu',padding = 'same')(tower1) # 5-size convolution
    tower2 = Conv1D(64,1,activation='relu',padding = 'same')(ins)    #
    tower2 = Conv1D(64,3,activation='relu',padding = 'same', dilation_rate = 3)(tower2) # 3-size convolution
    tower3 = Conv1D(64,1,activation='relu',padding = 'same')(ins)    # 1-size conolution
    tower4 = MaxPooling1D(2,padding='same')(ins)                     #
    tower4 = Conv1D(64,1,activation='relu',padding = 'same')(tower4) # Moxpool
    incept1 = concatenate([tower1, tower2, tower3, tower4], axis = 1)# Concatenate filters together
        
    pool1 =  MaxPooling1D(2)(incept1) #Maxpool everything
        
    tower5 = Conv1D(128,1,activation='relu',padding = 'same')(pool1) #
    tower5 = Conv1D(128,3,activation='relu',padding = 'same', dilation_rate = 3)(tower5)# 3-size convolution
    tower6 = MaxPooling1D(2,padding='same')(pool1)                   # 
    tower6 = Conv1D(128,1,activation='relu',padding = 'same')(tower6)# Maxpooling convolution
    incept2 = concatenate([tower5, tower6], axis = 1)                # Concatenate filter together
    
    pool2 =  MaxPooling1D(2)(incept2) #Maxpool everything
    
    tower7 = Conv1D(256,1,activation='relu',padding = 'same')(pool2) #
    tower7 = Conv1D(256,3,activation='relu',padding = 'same', dilation_rate = 3)(tower7)# 3-size convolution
    tower8 = MaxPooling1D(2,padding='same')(pool2)                   #
    tower8 = Conv1D(256,1,activation='relu',padding = 'same')(tower8)# Maxpooling convolution
    incept3 = concatenate([tower7, tower8], axis = 1)                # Concatenate filters together
    
    pool3 =  MaxPooling1D(2)(incept3) #Maxpool everything
    
    tower9 = Conv1D(256,1,activation='relu',padding = 'same')(pool3) # 
    tower9 = Conv1D(256,3,activation='relu',padding = 'same', dilation_rate = 3)(tower9)# 3-size convolution
    tower10 = MaxPooling1D(2,padding='same')(pool3)                  #
    tower10 = Conv1D(256,1,activation='relu',padding = 'same')(tower10) # Maxpool convolution
    incept4 = concatenate([tower9, tower10], axis = 1)               # Concatenate all filters together
    
    pool4 =  MaxPooling1D(2)(incept4) #Maxpool everything
    
    global_ = GlobalAveragePooling1D()(pool4)
    dense = Dense(labels.shape[1], activation = 'sigmoid')(global_)
        
    inceptor = Model(ins, dense, name = 'inceptor')
    inceptor.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        
    inceptor.summary()
    inceptor.fit(x,y, batch_size=32, epochs=100)
    os.chdir(r'C:\Users\isaac\Documents\Python Scripts')
    #inceptor.save('inception_reddit_text_300_300.h5')
       

    