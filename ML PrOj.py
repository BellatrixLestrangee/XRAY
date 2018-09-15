# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:16:19 2018

@author: UJJWAL
"""

#importing libraries
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import keras 
import tensorflow  
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Define path to the data directory
data_dir = Path('..\chest-xray-pneumonia\chest_xray')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir/'train'
# Path to validation directory
val_dir = data_dir/'val'

#test
normal_cases_test = val_dir/'NORMAL'
pneumonia_cases_test = val_dir/'PNEUMONIA'

# Get the list of all the images for test
normal_cases_val = normal_cases_test.glob('*.jpeg')
pneumonia_cases_val = pneumonia_cases_test.glob('*.jpeg')

test_data=[]
for img in normal_cases_val:
    test_data.append((img,0))
# for pneumonial case and value of those is 1
for img in pneumonia_cases_val:
    test_data.append((img,1))
    
test_data = pd.DataFrame(test_data, columns=['image','label'])

test_data = test_data.sample(frac=1.).reset_index(drop=True)

#Path to sub-director
normal_cases_dir = train_dir/'NORMAL'
pneumonia_cases_dir = train_dir/'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
# An empty list for  inserting new data
train_data = []
# for normal case and value of those is 0
for img in normal_cases:
    train_data.append((img,0))
# for pneumonial case and value of those is 1
for img in pneumonia_cases:
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'])

# Shuffle the data 
train_data = train_data.sample(frac=1.).reset_index(drop=True)

#length of images
n = len(train_data)

#creating a object
classifier = Sequential()
#convolution
classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3),activation='relu'))
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#use flatten functon
classifier.add(Flatten())
#full connection
classifier.add(Dense(output_dim=128,activation='relu'))#2nd layer
#classifier.add(Dense(output_dim=64,activation='relu'))#3rd layer
classifier.add(Dense(output_dim=1,activation='sigmoid'))#final
#complie
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the CNN into the images

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#trainig set
train_set = train_datagen.flow_from_directory('chest_xray/train' ,target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('chest_xray/val' ,target_size=(64,64),batch_size=32,class_mode='binary')

classifier.fit_generator(train_set,samples_per_epoch=n,nb_epoch=20,validation_data=test_set,nb_val_samples=16)

# For prediction
p = classifier.predict_generator(test_set,steps=1) 
prediction = pd.DataFrame(p) #
print(prediction)



