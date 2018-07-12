# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:10:23 2018

@author: Isaac Csekey
"""

#Text corpora, for extra reference when it comes to word2vec
from bs4 import BeautifulSoup as bs
from requests import get
import os
import urllib

#Making the directories

try:
    os.mkdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods\books")
except FileExistsError:
    pass
os.chdir(r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods\books")

#A main url, adding on the text files later
url = ['http://www.textfiles.com/etext/MODERN/','http://www.textfiles.com/etext/FICTION/','http://www.textfiles.com/etext/NONFICTION/']


#Getting the html from the main site
for entry in url:
    soup = bs(get(entry).text,'html.parser')

    #Luckily all hrefs on the site are of the .txt files
    a = soup.find_all("a")
    i = 0 #indexing for naming purposes
    #Going through all the links on the site, opening the files, writing them, and closing. Notice that not all files will be saved as .txt, for some reason
    for link in a:
    
        #Making sure the tag is a string
        print(type(link.text))
        string = link.text
    
        #Getting the .txt file online with the urllib package
        content=urllib.request.urlopen(entry+string)
    
        #Writing the file
        book = open(str(i)+link.text[0:4]+".txt", "w")
        for line in content:
            book.write(str(line))
            print(str(line))
        book.close()
        i+=1