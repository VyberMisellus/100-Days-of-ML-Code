# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:20:58 2018

@author: Isaac Csekey
"""

#MoodVec
#Word vectorizer that takes in the word files, reads them, and creates semantic vectors out of them


import nltk
import os
import re
import gensim.models as w2v
import multiprocessing
import codecs


#Getting the 'trash' character packages
nltk.download("punkt")
nltk.download("stopwords")

#The list of subreddit files
subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate','mentalhealth','depression_help','depressionregimens','Anxiety','books']
#The path that the files were saved to
path = r"C:\Users\Isaac Csekey\Documents\MoodData"

#Declaring an initial string for the sentences
all_subs = u""
#The function to read a list of files in each directory
def readPost(parent_directory, subreddit):
    
    #Returning a big ol string
    all_subs = u""
    
    for sub in subreddit:
        
        #Making a path according to the subreddit
        sub_path = os.path.join(path,sub)
        #Navigating to the directory
        os.chdir(sub_path)
    
        #Taking a look at all the files in the subreddit directory and reading them in
        for root, dirs, files in os.walk("."):  
            for filename in files:
                with codecs.open(filename, 'r', 'utf-8') as postfile:
                    try:
                        all_subs += postfile.read()
                    except UnicodeDecodeError:
                        pass
                    
    #Returning the lowercase words 
    return all_subs.lower()

#The function to tokenize and clean the sentences, returning a list of lists that can be fed into word2vec
def listOfWords(string):
    
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_raw = tokenizer.tokenize(string)
    
    #The sentences list
    s = []
    for sentence in sent_raw:
        if len(sentence)>0:
            clean = re.sub("[^a-zA-Z]"," ", sentence)
            s.append(clean.split())
    return s
    
    

#First getting the dictionary of raw data
all_subs = readPost(path,subreddits)

#Cleaning the data
sentences = listOfWords(all_subs)

#Creating the word2vec
sadvectors = w2v.Word2Vec(sentences, seed= 42, workers=multiprocessing.cpu_count(), size = 250, min_count = 3, window = 10, sample = 1e-3 )

#Training the word2vec neural net
sadvectors.train(sentences,total_examples=len(sentences),epochs=50)

os.chdir(r"C:\Users\Isaac Csekey\Documents\GitHub\100-Days-of-ML-Code")

#Saving the model
sadvectors.save('mood2vec.w2v')
