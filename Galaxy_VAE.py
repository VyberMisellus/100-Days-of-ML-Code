# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:44:52 2018

@author: Isaac Csekey

A variational convolutional-deconvolutional autoencoder for clustering images 
in a latent space representation. Applications include controllable image generation,
data compression, and image de-noising.

Inspired by https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Using data from the galaxy zoo challenge dataset: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
"""

import os
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import keras.backend as K
from keras.optimizers import RMSprop


IMAGE_HEIGHT = 88
IMAGE_WIDTH = 88
path_data = r"C:\Users\isaac\Pictures\Pythonic Data\Sample Galaxies\Samples"
os.chdir(path_data)
latent_num = 10



def grabIms(IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Load all images, reshaped using cropping 
    """
    print("Loading images...")
    #The directory for all the images in the database
    images_all = []    
    for root, dirs, files in os.walk("."):
            
    #Going through all the files 
        for filename in files:
        
            try:
                
                #Opening and cropping
                im = Image.open(filename)
                width, height = im.size   # Get dimensions
                left = (width - IMAGE_WIDTH)/2
                top = (height - IMAGE_HEIGHT)/2
                right = (width + IMAGE_WIDTH)/2
                bottom = (height + IMAGE_HEIGHT)/2
                
                im = im.crop((left, top, right, bottom))
                
                #Adding images to the list 
                images_all.append(im)
            except:
                pass
    



    #Creating one big array
    #        
    X = np.vstack(images_all).reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,3])
    X = X/np.max(X) #Scaling, just in case
    
    #Splitting into training and validation sets
    Xtr = X[0:int(0.8*len(X))]
    x_val = X[int(0.8*len(X)):]
    return Xtr, x_val

def show_gen_out(model,x_val,IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Plot the autoencoder's reconstruction of validation samples
    """
    
    print("Showing generated images")
    #Generator example outputs 
    sample = x_val
   
    for entry in sample: #Predicting for all given images
        prediction = model.predict(entry.reshape(-1,IMAGE_HEIGHT, IMAGE_WIDTH,3)).reshape(IMAGE_HEIGHT, IMAGE_WIDTH,3)
        plt.imshow(prediction)
        plt.show()
        
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
def loss(ytru, ypred):
    """
    A loss function composed of the sum of reconstruction loss, as well as of the kullback-leibler
    divergence to score the precision of the latent-space representations
    """
    reconstruction_loss = mse(K.flatten(ytru),K.flatten(ypred))
    reconstruction_loss*=pow(88,2)*3
    
    kl_loss = 1 + z - K.square(z) - K.exp(z)
    kl_loss = K.mean(kl_loss)
    kl_loss*=-0.5
    
    print('Making loss')
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    return vae_loss

    

def randomPlot(decoder,num_trials, rgb, latents, IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Plot random values in the latent space to demonstrate how effective of 
    a generator the decoder portion of the model is
    """
    
    channels = 1
    if rgb:
        channels = 3
    big = []
#Generate random images
    for i in range(num_trials):
        big.append(np.random.rand(latents))
    big = np.vstack(big)
    for entry in big:
        plt.imshow(decoder.predict(entry.reshape((-1,10))).reshape(IMAGE_HEIGHT, IMAGE_WIDTH,3))
        plt.show()
    return

with tf.device('/gpu:0'):
    ins= Input(shape = (IMAGE_HEIGHT, IMAGE_WIDTH,3))
#Creating the encoder
    #Adding Three convolutional and max-pooling layers for the encoder
    conv1 = Conv2D(64, (8,8), activation = 'relu', padding = 'same')(ins)
    max1 = MaxPooling2D((2,2))(conv1)#44x44x64
    conv2 = Conv2D(128, (4,4), activation = 'relu', padding = 'same')(max1)
    max2 = MaxPooling2D((2,2))(conv2)#128x22x22
    conv3 =Conv2D(256, (3,3), activation = 'relu', padding = 'same')(max2)
    max3 = MaxPooling2D((2,2))(conv3)#256x11x11

    flat = Flatten()(max3)
    dense1 = Dense(latent_num,activation = 'relu')(flat)
    
    #Sampling
    z_mean = Dense(latent_num, name='z_mean')(dense1)
    z_log_var = Dense(latent_num, name='z_log_var')(dense1)
    z = Lambda(sampling, output_shape = (latent_num,), name = 'z')([z_mean, z_log_var])
    
    encoder = Model(ins, z, name = 'encoder')
    encoder.summary()
    
#Creating the decoder        
    latent_ins = Input(shape = (latent_num,))
    dense2 = Dense(11*11*256)(latent_ins)
    resh = Reshape((11,11,256))(dense2)
        
    #Three deconvolution layers with max-pooling for each
    decon1 = Conv2D(256,(3,3), activation = 'relu', padding='same')(resh)
    ups1 = UpSampling2D((2,2))(decon1)#22x22x256
    decon2 = Conv2D(128, (4,4), activation = 'relu', padding='same')(ups1)
    ups2 = UpSampling2D()(decon2)#128x44x44
    decon3 = Conv2D(64, (8,8), activation = 'relu', padding='same')(ups2)
    ups3 = UpSampling2D()(decon3)#64x88x88
    Outs = Conv2D(3, (3,3), activation = 'relu', padding='same')(ups3)
    decoder = Model(latent_ins,Outs)
    
#Full VAE construction 
    outputs = decoder(encoder(ins))
    vae = Model(ins, outputs)
    vae.compile(optimizer = RMSprop(lr = 0.0001), loss = loss) #Modified learning rate due to sensitivity of the model 
    vae.summary()
    
#Training    
if __name__ == '__main__':
    os.chdir(path_data)
    Xtr, x_val = grabIms(IMAGE_HEIGHT, IMAGE_WIDTH) 
    
    for i in range(10):
        vae.fit(Xtr, Xtr,epochs=20,batch_size=64,validation_data=(x_val, x_val), verbose = 1)
        show_gen_out(vae, x_val[:10], IMAGE_HEIGHT, IMAGE_WIDTH)
        randomPlot(decoder, num_trials = 10,latents = latent_num, rgb = True, IMAGE_HEIGHT=IMAGE_HEIGHT, IMAGE_WIDTH=IMAGE_WIDTH )
        