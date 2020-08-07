#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:02:22 2020

@author: Aron
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import seaborn as sns
import cv2
from random import randint
import shutil


os.getcwd()
os.chdir('/Users/Aron/Kaggle/plant_pathology')
local_dir = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7'
kaggle_dir = '../input/plant-pathology-2020-fgvc7/'

sample_submission = pd.read_csv(local_dir+'/sample_submission.csv')
test = pd.read_csv(local_dir+'/test.csv')
train = pd.read_csv(local_dir+'/train.csv')


# separate the training data into its own folders

#grab all of the trianing images into a list


multi = train[train['multiple_diseases'] == 1]
multi = multi['image_id']
healthy = train[train['healthy'] == 1]
healthy = healthy['image_id']
rust = train[train['rust'] == 1]
rust = rust['image_id']
scab = train[train['scab'] == 1]
scab = scab['image_id']


src = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images'
multi_dest = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images/multi'
healthy_dest = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images/healthy'
rust_dest = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images/rust'
scab_dest = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images/scab/'
test_dest = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7/images/test'

#move the multi images
for filename in multi:
    shutil.move(os.path.join(src, filename+'.jpg'),os.path.join(multi_dest, filename+'.jpg'))
#move the helthy images
for filename in healthy:
    shutil.move(os.path.join(src, filename+'.jpg'),os.path.join(healthy_dest, filename+'.jpg'))
#move the rust images
for filename in rust:
    shutil.move(os.path.join(src, filename+'.jpg'),os.path.join(rust_dest, filename+'.jpg'))
# move the scab images
for filename in scab:
    shutil.move(os.path.join(src, filename+'.jpg'),os.path.join(scab_dest, filename+'.jpg'))
#move the test images
test_images = [ file for file in os.listdir(src) if file.startswith('Test_') ] 

for filename in test_images:
    shutil.move(os.path.join(src, filename),os.path.join(test_dest, filename))
    


#Now lets set up the image generator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

img = load_img(scab_dest+scab[0]+'.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape) 

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=src+'/preview', save_prefix='scabs', save_format='jpeg'):
    i += 1
    if i > 20:
        break
        
        
        
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))    
        
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=.001),
              metrics=['accuracy'])       



test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
batch_size = 8
train_generator = datagen.flow_from_directory(
        src+'/train',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'training')  

validation_generator = datagen.flow_from_directory(
        src+'/train',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'validation')


model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')








