# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:34:51 2024

@author: Gaurav
"""

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import BatchNormalization, Layer, Flatten, Dense, Input, Reshape, Dropout
from keras.activations import sigmoid, tanh, relu
from keras.layers import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy


image_rows = 28
image_cols = 28
channels = 1
img_shape = (image_rows, image_cols, channels)

def build_generator():
    
    noise_shape = (100, )
    model = Sequential()
    
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh')) #-1 to 1 prrediction
    model.add(Reshape(img_shape))#1D array ko 28, 28, 1 ki shape me reshape karne ke liye
    
    model.summary()
    noise = Input(noise_shape) #noise dene ke liye input layer
    img = model(noise) #output after taking input
    
    return Model(noise, img) #ye bata raha hai ye noise lega image dega

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

def train(epochs, batch_size=128, save_interval=500):
    (x_train, _), (_, _) = mnist.load_data()
    
    x_train = (x_train.astype(np.float32) - 127.5)/127.5 #scale -1 to 1 for tanh function
    
    x_train = np.expand_dims(x_train, axis=3)
    
    half_batch = int(batch_size/2)
    
    for epochs in range(epochs):
        
        #train Discriminator
        
        idx = np.random.randint(0, x_train.shape[0], half_batch) 
        img = x_train[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        #0 is the mean and 1 is SD meaning -1 to 1
        
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(img, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
        
        #train Generator
        
        noise = np.random.normal(0, 1, (half_batch, 100))
        y_valid = np.array([1]*half_batch)
        
        g_loss = combined.train_on_batch(noise, y_valid)
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epochs, d_loss[0], 100*d_loss[1], g_loss))
        
        if epochs % save_interval == 0:
            save_imgs(epochs)
            
def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r*c, 100))
    gen_imgs = generator.predict(noise)
    
    #Rescale images -1 to 1
    gen_imgs = 0.5*gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()

optimizer_gen = Adam(0.0002, 0.5)
optimizer_disc = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(optimizer=optimizer_disc, loss = 'binary_crossentropy', metrics=['accuracy'])

generator = build_generator()
generator.compile(optimizer=optimizer_gen, loss='binary_crossentropy')

z = Input(shape=(100,))
img = generator(z)

discriminator.trainable = False

valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

train(epochs=10000, batch_size=32, save_interval=1000)