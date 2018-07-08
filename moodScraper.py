# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:17:47 2018

@author: Isaac Csekey
"""
#Import PRAW for scraping reddit (obviously)
import praw
#OS for creating save directories to pull from later, where subreddit posts/comments will be saved based on their names
import os


#A post-grabbing function, where it will accept a string that represents a subreddit name
def postGrab(subreddit_name):

    try:
        os.mkdir(os.path.join(r"C:\Users\moodScraper",subreddit_name))

    #If the directory already exists for whatever reason
    except FileExistsError:
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

            #Printing to know it works, and to partly feel like Hackerman
            print(submission.selftext)
            print(all_.body)
        
            #Tagging each comment by putting "</(COMMENT)\>" at the beginning, so that when the file is later parsed by another program, it's easier to identify a comment apart from a main post
            try:
                fd.write("</(COMMENT)\>"+all_.body)
            #More potentially pesky emojis
            except UnicodeEncodeError:
                pass
            except PermissionError:
                pass
        fd.close()

#MAKE SURE TO GO TO REDDIT'S DEV SITE TO REGISTER A USER AGENT
redd = praw.Reddit(client_id=, client_secret=, user_agent=, username=, password=)

#A list of the subreddits being used, feel free to add as many as available. This scraper is meant to get all the "moods," but it works for all subreddits
subreddit_list = ['angry',"SuicideWatch",'depression','happy','BPD','mentalillness','sad','hate']

#Using the function on each subreddit
for sub in subreddit_list:
    postGrab(sub)
