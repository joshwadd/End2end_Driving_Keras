import csv
import cv2
import numpy as np
import os
import glob
from random import shuffle

os.environ["KERAS_BACKEND"] = "tensorflow"
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

##Open all the recorded run directorys
data_runs_train = glob.glob('/media/josh/DATADRIVE2/AutomousCarData/training/*')
data_runs_valid = glob.glob('/media/josh/DATADRIVE2/AutomousCarData/validation/*')

## Read the samples from the training data
## Load data into list of lists
train_samples = []
for i, run_dir in enumerate(data_runs_train):
    lines = []
    with open(run_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            train_samples.append(line)

shuffle(train_samples)
total_train_samples = len(train_samples)
## Read the samples from the validation data
## Load data into list of lists
validation_samples = []
for i, run_dir in enumerate(data_runs_valid):
    lines = []
    with open(run_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            validation_samples.append(line)

total_valid_samples = len(validation_samples)
shuffle(validation_samples)

print('Data set is made up of .....')
print(total_train_samples, ' training samples ')
print(total_valid_samples, ' validation samples ')

drop_prob = 0.2
learning_rate = 0.0001
batch_size = 256
seq_len = 1
epochs = 30

steering_correction = 0.2
lag = 2

#####################################
## Image generators and preprocessing
####################################

def choose_camera(center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return cv2.imread(left), steering_angle + steering_correction
    elif choice ==1:
        return cv2.imread(right), steering_angle - steering_correction

    return cv2.imread(center), steering_angle

def generator(samples, batch_size = 32):
    while 1:

        X = []
        y = []

        i =0
        for ix in np.random.permutation(len(samples)):
            sample = samples[ix]
            center_path = sample[0]
            left_path = sample[1]
            right_path = sample[2]
            steering_angle = float(sample[3])

            image, steering_angle = choose_camera(center_path, left_path, right_path, steering_angle)
            X.append(image)
            y.append(steering_angle)
            i +=1
            if i == batch_size:
                break

        yield np.array(X), np.array(y)

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

##Time distributed Nvidia CNN style model
model.add(Lambda(lambda x: x/127.5 -1.0, input_shape=( 160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5, 5) , activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(36, (5, 5) , activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(48, (3, 3) , activation='relu', strides=(2,2), padding='valid'))
model.add(Conv2D(64, (3, 3) , activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add((Dropout(drop_prob)))
model.add(Flatten())
##LSTM Componen
model.add(Dense(100, activation='relu'))
model.add(Dropout(drop_prob))
model.add(Dense(50, activation='relu'))
model.add(Dropout(drop_prob))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

check_point = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0,
                              save_best_only=True, mode='auto')


adam_op = optimizers.Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=adam_op)

model.fit_generator(generator = train_generator, steps_per_epoch = total_train_samples // batch_size, validation_data = validation_generator,
                    validation_steps = (total_valid_samples // batch_size), epochs=epochs, shuffle=True, callbacks=[check_point])

#model.save('model.h5')





