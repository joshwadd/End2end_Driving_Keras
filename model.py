import csv
import cv2
import numpy as np
import os
import glob
from random import shuffle

import matplotlib.image as mpimg
import random
import skimage.transform as sktrans
from sklearn.model_selection import train_test_split

from data import *


##################################
### LOAD DATA AND BUILD GENERATORS
##################################
## Load the assoiated image and control data for each entry
## Note the layout of each line in the csv is as follows
##line[0] : center image
##line[1] : left image
##line[2] : right image
##line[3] : steering angle
##line[4] : throttle
##line[5] : break
##line[6] : speed


data_path = '/media/josh/DATADRIVE2/AutomousCarData2/driving_log_balanced.csv'

## Read the samples from the training data
## Load data into list of lists
train_samples = []
with open(data_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        train_samples.append(line)

shuffle(train_samples)


train_samples , validation_samples = train_test_split(train_samples, test_size=0.2)

total_valid_samples = len(validation_samples)
total_train_samples = len(train_samples)


drop_prob = 0.5
learning_rate = 0.00005
batch_size = 40
epochs = 10
steering_correction = 0.25

training_image_size = (66, 200, 3)


#Create the generator and validation generators
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


####################################
## BUILD LSTM + CNN MODEL
###################################

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense, Cropping2D
from keras.layers.wrappers import  TimeDistributed
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint


model = Sequential()

##Nvidia CNN style model
model.add(Lambda(lambda x: x/127.5 -1.0, input_shape=training_image_size) )
#model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5, 5) , activation='elu', strides=(2,2), padding='valid'))
model.add(Conv2D(36, (5, 5) , activation='elu', strides=(2,2), padding='valid'))
model.add(Conv2D(48, (3, 3) , activation='elu', strides=(2,2), padding='valid'))
model.add(Conv2D(64, (3, 3) , activation='elu'))
model.add(Conv2D(64, (3, 3) , activation='elu'))
model.add((Dropout(drop_prob)))
model.add(Flatten())
##Fully connected layers
model.add(Dense(100, activation='elu'))
model.add(Dropout(drop_prob))
model.add(Dense(50, activation='elu'))
model.add(Dropout(drop_prob))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

check_point = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0,
                              save_best_only=True, mode='auto')


adam_op = optimizers.Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=adam_op)

model.fit_generator(generator = train_generator, steps_per_epoch = total_train_samples // batch_size, validation_data = validation_generator,
                    validation_steps = (total_valid_samples // batch_size), epochs=epochs, shuffle=True, callbacks=[check_point])

#model.save('model.h5')





