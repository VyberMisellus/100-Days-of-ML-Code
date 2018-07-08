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


#Getting the 'trash' character packages
nltk.download("punkt")
nltk.download("stopwords")

#The list of subreddit files
subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate']
#The path that the files were saved to
path = r"C:\Users\Isaac Csekey\Documents\MoodData"
all_subs = dict()
print(type(all_subs))

#The function to read a list of files in each directory
def readPost(parent_directory, subreddit):
    
    #What will be returned: A big dictionary!
    all_subs = dict()
    print(type(all_subs))
    
    for sub in subreddit:
        
        #Making a path according to the subreddit
        sub_path = os.path.join(path,sub)
        #Navigating to the directory
        os.chdir(sub_path)
    
        #Making an empty list of all the posts in the one subreddit
        all_posts = []
    
        #Taking a look at all the files in the subreddit directory and reading them in
        for root, dirs, files in os.walk("."):  
            for filename in files:
                tempfile = open(str(filename),"r")
                tempfile = str(tempfile.read())
            
                tempfile = tempfile.split("</(COMMENT)\>")
                all_posts.append(tempfile)
                
        all_subs[sub] = all_posts
        print(type(all_subs))
    return all_subs

#The function to convert all the posts and comments in the dictionaries to a list of lists of strings
def toBigStrings(dictionary_of_subs):
    
    #Initialize the list variable
    biglist = []
    
    #Chunking through each category in the dictionary
    for entry in dictionary_of_subs:
        category = dictionary_of_subs[entry]
        
        #Chunking through each list in the category list of lists
        for entries in category:
            #Chunking through the list of entries lists
            for ent in entries:
                #Some 'preprocessing' to get only the words
                clean = re.sub("[^a-zA-Z]"," ", ent)
                #appending the list of strings given the cleaned data
                biglist.append(clean)
    return biglist

print(type(all_subs))

#First getting the dictionary of raw data
all_subs = readPost(path,subreddits)

#Cleaning up and assigning a big list of words
bigfile = toBigStrings(all_subs)