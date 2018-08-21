# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:56:23 2018

@author: isaac
"""
import numpy as np
import pickle
import os
import time


path = r"C:\Users\isaac\Documents\GitHub\100-Days-of-ML-Code\moods"
os.chdir(path)
print("Opening pickle file")
with open("vectorizedsentences.txt",'rb') as fp:
    x = pickle.load(fp)

formatted_x = []
print("Converting to long...")

#@vectorize(['float32(float32,float32)'], target = 'gpu')
def matricer(x):
    formatted_x = []
    i=1
    for matrix in x:
        print("Converting matrix ",i," out of ",len(x))
        i+=1
    
        for vector in matrix:
            try:
        
                for entry in vector:
            
                    formatted_x.append(entry)
                    #print("Conversion success.")
            except TypeError:
            #print("Error passed")
                pass
        time.sleep(0.01)
#    longm.append(longv)
    vector_of_all = np.array(formatted_x).reshape([-1,300,300,1])
    print("Done")

    np.save("vectored_data.npy",vector_of_all)
    print("File saved.")
    return vector_of_all
ms = matricer(x)
print(ms.shape)