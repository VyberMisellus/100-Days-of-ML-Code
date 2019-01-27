# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:17:47 2018

@author: Isaac Csekey
"""

#This file uses Reddit's PRAW api to scrape posts from different specified subreddits on the web. 
#It cleans any emoticons and extracts raw text from both posts and comments in each subreddit
#This program demonstrates a very basic data extraction and cleaning process, where the raw data
#is scraped from forums, and valuable text and information on the responses (such as if the text is a comment or not)
#Is kept. 

#The possibilities of what could be modeled or discovered from the data are potentially endless, and all tie back to 
#the field of Natural Language processing (NLP). From automated question-answering to semantically-driven sentiment analysis,
#this type of 'chat' data has myriad uses for modelling the technologies of tomorrow.

#Import PRAW for scraping reddit 
import praw
#OS for creating save directories to pull from later, where subreddit posts/comments will be saved based on the forum names
import os


#A post-grabbing function, where it will accept a string that represents a subreddit name
def postGrab(subreddit_name):

    try:
        os.mkdir(os.path.join(r"C:\Users\moodScraper",subreddit_name))

    #If the directory already exists for whatever reason
    except FileExistsError:
        print('Directory already exists')
        pass
    
    finally:
        os.chdir(os.path.join(r"C:\Users\moodScraper",subreddit_name))
        pass

    #Creating a temporary subreddit instance using the string passed into postGrab
    temp_sub = redd.subreddit(subreddit_name)

    #Chunking through each post, and writing the original post and all the comments to a .txt file to the associated directory
    for submission in temp_sub.hot(limit = None):

        #Creating the file, named after the submission ID
        fd = open(str(submission.id)+ ".txt","w")

        #Making sure all the characters are unicode-writeable, maybe some emojis aren't recognized
        try:
            fd.write(submission.selftext)
        except UnicodeEncodeError:
            pass

        #Chunking through each comment forest    
        for all_ in submission.comments:
            if isinstance(all_, praw.models.MoreComments):
                continue
        
            #Tagging each comment by putting "</(COMMENT)\>" at the beginning, so that when the file is later parsed by 
            #another program, it's easier to identify a comment apart from a main post
            try:
                fd.write("</(COMMENT)\>"+all_.body)
            #More potentially pesky emojis
            except UnicodeEncodeError:
                pass
            except PermissionError:
                pass
        fd.close()

#MAKE SURE TO GO TO REDDIT'S DEV SITE FOR API ACCOUNT
redd = praw.Reddit(client_id=, client_secret=, user_agent=, username=, password=)

#A list of the subreddits being used, feel free to add as many as available. This scraper is meant to get all the "moods," but it works for all subreddits
subreddit_list = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate']

#Using the function on each subreddit
for sub in subreddit_list:
    postGrab(sub)
