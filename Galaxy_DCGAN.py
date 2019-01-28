# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:01:47 2018

@author: isaac

An experimental DCGAN from the ground-up
"""

#Using a deep convolutional and deconvolutional neural network, the DCGAN pits 
#the two networks against each other: The convolutional network predicts whether
#or not the images given to it are generated, and the deconvolutional network 
#tries to learn how to generate images to 'fool' the convolutional network. 
#Traditionally, the convolutional network acts as a 'discriminator' and the 
#deconvolutional one the 'generator'

#For the sake of simplicity, the data used is a uniform set of small images from 
#Galaxy Zoo dataset
import os
from skimage.transform import resize
from matplotlib import pyplot
import helper
from scipy import misc
import numpy as np
import tensorflow as tf
from matplotlib.image import imread
from PIL import Image
path_originals = r"C:\Users\isaac\Pictures\Pythonic Data\Sample Galaxies\Samples"
path_generated =  r"C:\Users\isaac\Pictures\Pythonic Data\Sample Galaxies\Want"



#Get on with the sinful first
print("Getting the sinful anime")
os.chdir(path_originals)

IMAGE_HEIGHT = 88
IMAGE_WIDTH = 88
   
#Navigate to the directory for all the images in the database and extract them
galaxy_all = []    
for root, dirs, files in os.walk("."):
            
    #Going through all the files 
    for filename in files:
        print("Reading ",filename)
        
        try:
            temp = misc.imread(filename,mode="RGB")
        
            temp = resize(temp, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True)
            galaxy_all.append(temp)
        except:
            pass

#Creating one big array 
X = np.vstack(galaxy_all).reshape([-1,IMAGE_HEIGHT,IMAGE_WIDTH,3])
print("Array saved")
shape = len(X), IMAGE_HEIGHT,IMAGE_WIDTH, 3 
#________________________________________________________________GETTING IMAGES
def get_batches(batch_size, x):
    """
    Generate batches
    """
    
    with tf.device('/gpu:0'):
        IMAGE_MAX_VALUE = 255

        print("Getting batches")
        current_index = 0
        while current_index + batch_size <= shape[0]:
            data_batch = x[current_index:current_index + batch_size]
            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5

#________________________________________________________________________INPUTS

#Defining inputs first

def model_inputs(image_width, image_height, image_channels, z_dim):
    print("Creating model inputs")
    #This is the first function to be used, as it creates all the variables to be used in the network
    #First, a 32-bit floating point variable that is a 4-D tensor, named inputs_real
    with tf.device('/gpu:0'):
        inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name = 'input_real')
    #Then, the randomized inputs for the generator, a matrix with vector of random numbers for each image, where the rows are unique to an image
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name= 'input_z')
    #Then, defining the learning rate for the networks
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #Returning the tensorflow placeholder variables to be declared later on
    return inputs_real, inputs_z, learning_rate


#______________________________________________________________________NETWORKS

#Defining the discriminator network, the 'art critic', made up of a convnet, batch normalization for speed, then a leaky ReLU
def discriminator(images, reuse = False):
    alpha = 0.2 #A hyperparameter defining the 'shrinking factor' of entries between two matrices
    #Instead of setting all negative entries in a matrix to zero, a leaky relu will return negative entries that are a tenth of what they originally were,
    #still preserving negative entries in the image
    
    #With a bunch of variables, it would suck to have to get them all individually, rather 
    #the fact that the variables are declared under an umbrella 'scope' of 'discriminator'
    print("Initialize discriminator")
    with tf.variable_scope('discriminator', reuse = reuse):
        #A 4-layer network, hoh boy
        with tf.device('/gpu:0'):
            conv0 = tf.layers.conv2d(images, 32, 10, 2, 'SAME') #64 individual convolutional filters, filter size of 5, and a pixel stride of 2
            leakReLU0 = tf.maximum(alpha*conv0, conv0)
        
        
        #Conv1
            conv1 = tf.layers.conv2d(leakReLU0, 64, 5, 2, 'SAME') #64 individual convolutional filters, filter size of 5, and a pixel stride of 2
            leakReLU1 = tf.maximum(alpha*conv1, conv1) #Getting the shrunken negative values of the convolved outputs
        
        #Conv 2
            conv2 = tf.layers.conv2d(leakReLU1, 128, 5, 2, 'SAME') #More convolutional layers, 128 to be exact, passed from the previous layer, see the leakReLU
            batch_norm2 = tf.layers.batch_normalization(conv2, training=True) #Batch normalization, returning the output in training mode, used to normalize all outputs to keep the training 'stable' and less varied
            leakReLU2 = tf.maximum(alpha * batch_norm2, batch_norm2)  #Another leaky ReLU unit
        
        #Conv 3, a repetition of conv2 with more filters an smaller stride
            conv3 = tf.layers.conv2d(leakReLU2, 256, 5, 2, 'SAME')
            batch_norm3 = tf.layers.batch_normalization(conv3, training = True)
            leakReLU3 = tf.maximum(alpha*batch_norm3,batch_norm3)
##        
##        #Conv 4
            conv4 = tf.layers.conv2d(leakReLU3, 512, 5, 2, 'SAME')
            batch_norm4 = tf.layers.batch_normalization(conv4, training = True)
            leakReLU4 = tf.maximum(alpha*batch_norm4,batch_norm4)
#            
#            conv5 = tf.layers.conv2d(leakReLU4, 1024, 5, 2, 'SAME')
#            batch_norm5 = tf.layers.batch_normalization(conv5, training = True)
#            leakReLU5 = tf.maximum(alpha*batch_norm5,batch_norm5)
        
        #Flattening the outputs
            flat = tf.reshape(leakReLU4, (-1,4*4*512)) #Taking the output of each image passed through and creating a vector from each image to be passed through a fully-connected layer
        
        #Fully connected layer, a 'logit' that will give the probability of the image being real or fake, real being an output near 1, fake being near 0
            full_con = tf.layers.dense(flat, 1)
        
        #The activation function for this fella will be a sigmoid function
            out = tf.sigmoid(full_con)
            print("Returning discriminator outputs")
        return out, full_con
    
#Defining the generator network: Almost identical to the discriminator, however it deconvolves instead of convolves     
def generator(z, out_channel_dim, is_train = True):
    
    print("Generating outputs")
    alpha = 0.2 #Another hyperparameter, same as the discriminator
    with tf.variable_scope('generator',reuse = False if is_train ==True else True):
        with tf.device('/gpu:0'):
        #Fully connected layer
            full_con_1 = tf.layers.dense(z, 4*4*512) 
        
        #Reshape the fully connected layer for convolutional layer compatability
            deconv_2 = tf.reshape(full_con_1, (-1,4,4,512)) #Creating the n-number of 2 x 2 x 512 filters for n images
            batch_norm2 = tf.layers.batch_normalization(deconv_2, training = is_train) #feeding the deconvolutional layer into some batch-normalization
            leakReLU2 = tf.maximum(alpha*batch_norm2, batch_norm2)
        
        #Deconv 1
            deconv3 = tf.layers.conv2d_transpose(leakReLU2, 256, 5, 2, padding = 'VALID') #Deconvolve the image 
            batch_norm3 = tf.layers.batch_normalization(deconv3, training = is_train)
            leakReLU3 = tf.maximum(alpha*batch_norm3, batch_norm3)
        
            # Deconv 2
            deconv4 = tf.layers.conv2d_transpose(leakReLU3, 128, 5, 2, padding='SAME')
            batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
            leakReLU4 = tf.maximum(alpha * batch_norm4, batch_norm4)
            
        #Deconv 3
            deconv5 = tf.layers.conv2d_transpose(leakReLU4, 64, 5, 2, padding='SAME')
            batch_norm5 = tf.layers.batch_normalization(deconv5, training=is_train)
            leakReLU5 = tf.maximum(alpha * batch_norm5, batch_norm5)
##        
##        #Deconv 4
#            deconv6 = tf.layers.conv2d_transpose(leakReLU5, 32, 5, 2, padding='SAME')
#            batch_norm6 = tf.layers.batch_normalization(deconv6, training=is_train)
#            leakReLU6 = tf.maximum(alpha * batch_norm6, batch_norm6)
            
        #Deconv 5
#            deconv7 = tf.layers.conv2d_transpose(leakReLU6, 16, 5, 2, padding='SAME')
#            batch_norm7 = tf.layers.batch_normalization(deconv7, training=is_train)
#            leakReLU7 = tf.maximum(alpha * batch_norm7, batch_norm7)
        
        #Output layer
            logits = tf.layers.conv2d_transpose(leakReLU5, out_channel_dim, 5,2,padding = 'SAME') #More image transpose
        
            out = tf.tanh(logits)
            print("Out-channel is ",out.shape)  
        return out

#________________________________________________________________LOSS FUNCTIONS
        
#Three loss functions are needed: generator lossm, and the discriminator true and fake image loss
        
def model_loss(input_real, input_z, out_channel_dim):
    #Label smoothing hyperparameter
    label_smoothing = 0.9
    print("Calculating loss")
    with tf.device('/gpu:0'):
        gen_model = generator(input_z, out_channel_dim) #Generating images
        discrim_model_real, discrim_logits_real = discriminator(input_real) #Getting the scores of real images
        discrim_model_fake, discrim_logits_fake = discriminator(gen_model, reuse = True) #Seeing how the generated images perform on the discriminator
        
        discrim_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discrim_logits_real, labels = tf.ones_like(discrim_model_real )*label_smoothing))
        discrim_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discrim_logits_fake,labels=tf.zeros_like(discrim_model_fake)))
        
        discrim_loss = discrim_loss_real+discrim_loss_fake
    
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= discrim_logits_fake, labels = tf.ones_like(discrim_model_fake)*label_smoothing))
    
    return discrim_loss, gen_loss

#_____________________________________________________________________OPTIMIZER
def optimizer(d_loss, g_loss, learn_rate, beta1):
    print("Optimizing")
    #Initializing an instance of trainable variables
    
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
    #Optimizing
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.device('/gpu:0'):
            d_train_opt = tf.train.AdamOptimizer(learn_rate, beta1=beta1).minimize(d_loss, var_list = d_vars)
            g_train_opt = tf.train.AdamOptimizer(learn_rate, beta1=beta1).minimize(g_loss, var_list = g_vars)
    

    #Returning optimized variable    
    return d_train_opt, g_train_opt

#_________________________________________________________________VISUALIZATION
def show_gen_out(sess,n_images,input_z,out_channel_dim, output,name):
    print("Showing generated images")
    os.chdir(path_generated)
    #Generator example outputs 
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1,1,size=[n_images, z_dim])
    
    samples = sess.run(generator(input_z,out_channel_dim, False), feed_dict={input_z: example_z})
    print(samples[0].shape)
    misc.imsave(str(name)+".jpg",samples[0])
    if output:
        pyplot.imshow(samples[0])
        pyplot.show()
    
#______________________________________________________________________TRAINING
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches,data_shape, x):
    print("Training")
    #TRAINING THE ENTIRE DCGAN
    
    #Starting witht the variables
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = optimizer(d_loss,g_loss, learning_rate, beta1)
    steps = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size, x):
                
                #values range from -0.5 to 0.5, therefore scale to range -1, 1
                batch_images = batch_images*2
                
                steps+=1
                
                batch_z = np.random.uniform(-1,1,size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                if True:
                    if steps % 500 == 0:
                        show_im = True
                        _ = show_gen_out(sess, 1, input_z, data_shape[3], show_im, steps)
                    else:
                        show_im = False
                    # At the end of every 10 epochs, get the losses and print them out
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    
                    
batch_size = 64
z_dim = 1000
learning_rate = 0.0001
beta1 = 0.5
epochs = 1000000
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, shape, X)
                