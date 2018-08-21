# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 07:26:14 2018

@author: isaac
"""

#Read in Data and reformat it into the proper vector shape
import gensim.models as w2v
import os
import re
import codecs
import pickle
os.chdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code")

#Loading a google word vectors library
svctrs= w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary= True)
#Setting up training and testing data 

labelslist = subreddits = ['angry',"SuicideWatch",'depression','happy','BPD','sad','hate']
path = r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods"
#data = [] #For training and testing

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
                try:
                    with codecs.open(filename, 'r', 'utf-8') as file:
                        try:
                            collectionstring += file.read()
                        except UnicodeDecodeError:
                            pass
                except FileNotFoundError:
                    pass
                
                finally:
                    file.close()
                print("Opened ",filename)
                
                #Cleaning the data a little
                collectionstring = re.sub("[^a-zA-Z]"," ", collectionstring)
                
                collectionstring = collectionstring.lower()
                
                collectionstring = collectionstring.split()
                
                
                #Converting strings to their embedded vectors from the pre-trained word2vec model
                
                
                
                category_list.append(collectionstring)
                
        labelled_data[item] = category_list
        print(item, "loaded to dictionary\n")
        
    del(collectionstring)
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
    total_Y = x= []
    data_conv = dict()
    i = 0
    for item in labelslist:
        temp = [0]*len(labelslist)
        temp[i] = 1
        data_conv[item] = temp
        i+=1
    
    #Through each key
    for key in dataDictionary.keys():
        print("Starting on ",key)
        
        #Each set of posts for each category
        post_set = dataDictionary[key]
        
        i = 1
        #Going through each post
        for post in post_set:
            print("On post ",i,"/",len(post_set))
            #Making an embedding matrix from each string
            matrix_embed = []
            
            #Going through each word in the post
            for word in post:
                
                #Will only give embeddings for words in the embedding dictionary
                try:
                    matrix_embed.append(svctrs[word].tolist())
                except KeyError:
                    pass
            if len(matrix_embed) > 300:
                matrix_embed = matrix_embed[:299]
            if len(matrix_embed)>0:
                #Defining the number of extra padded vectors for the embedding matrix
                length = 300-len(matrix_embed)
                j = 0
                
                #Padding the matrix
                while j < length:    
                    matrix_embed.append([0.0]*300) 
                    j+=1
            
                x.append(matrix_embed)   
                
                total_Y.append(data_conv[key])
            i+=1
            #time.sleep(0.1)
    return x, total_Y

        
data_dict = getData(labelslist)

roughx, y = vectorize(data_dict)
print("Deleting the google vector library")
del(svctrs,data_dict)
print("Writing to pickled file.")
os.chdir(path)
with open("vectorizedsentences.txt", "wb") as fp:   #Pickling
    pickle.dump(roughx, fp)
with open("labels.txt","wb") as dd:
    pickle.dump(y,dd)