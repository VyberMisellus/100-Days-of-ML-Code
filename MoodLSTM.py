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
import pandas

os.chdir(r"C:\Users\Isaac Csekey\Documents\MoodData")
svctrs = w2v.KeyedVectors.load("mood2vec.w2v")

