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

os.chdir(r"C:\Users\Isaac Csekey\Documents\MoodData")
svctrs = w2v.KeyedVectors.load("mood2vec.w2v")

#Setting up training and testing data 

labelslist = subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate','mentalhealth','depression_help','depressionregimens','Anxiety']
path = r"C:\Users\Isaac Csekey\Documents\MoodData"


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
                file = open(filename, 'r')
                for line in file:
                    collectionstring += line
                
                print("Opened ",filename)
                
                
                collectionstring = re.sub("[^a-zA-Z]"," ", collectionstring)
                
                category_list.append(collectionstring.lower())
                
        labelled_data[item] = category_list
        print(item, "loaded to dictionary\n")
        
    return labelled_data

data_dict = getData(labelslist)
