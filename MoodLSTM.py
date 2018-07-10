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
from autocorrect import spell

os.chdir(r"C:\Users\Isaac Csekey\Documents\GitHub\100-Days-of-ML-Code")
svctrs = w2v.KeyedVectors.load("mood2vec.w2v")

#Setting up training and testing data 

labelslist = subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate','mentalhealth','depression_help','depressionregimens','Anxiety']
path = r"C:\Users\Isaac Csekey\Documents\MoodData"
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
                
                file = open(filename, 'r')
                for line in file:
                    collectionstring += line
                
                print("Opened ",filename)
                
                #Cleaning the data a little
                collectionstring = re.sub("[^a-zA-Z]"," ", collectionstring)
                
                collectionstring = collectionstring.lower()
                
                collectionstring = collectionstring.split()
                
                
                #Converting strings to their embedded vectors from the pre-trained word2vec model
                for item in collectionstring:
                    
                    try:
                        temp = spell(item)
                        embed.append(svctrs[str(temp)])
                    except KeyError:
                        print(item,temp)
                print("Completed embeddings for", filename)
                
                
                category_list.append(embed)
                
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

data_dict = getData(labelslist)

XY = toXY(data_dict)

#Randomizing the entries in the list 
   
data = XY
shuffle(data)


#This is where the fun begins
#______________________________________________________________________________
#|                               LSTM TIME                                    | 
#|____________________________________________________________________________|

