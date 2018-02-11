import csv
import cv2
import numpy as np
import os
import glob
from random import shuffle

import matplotlib.image as mpimg
import random
import skimage.transform as sktrans


training_image_size = (66, 200, 3)
steering_correction = 0.25

#####################################
## Image generators and preprocessing
####################################

def preprocess(image, color_tran=True):
    image = crop_image(image)
    image = resize_image(image)
    if color_tran:
        image = colour_space_transform(image)


    return image

def augment_data(center_path, left_path, right_path, steering_angle):
    image, steering_angle = choose_camera(center_path, left_path, right_path, steering_angle)
    image, steering_angle = flip_image(image, steering_angle)
    image, steering_angle = translate_image(image, steering_angle, 100, 10)
    image = add_shadow(image)
    #image = change_brightness(image)

    return image, steering_angle

def translate_image(image, steering_angle, x_range, y_range):

    translate_x = x_range * (np.random.rand() - 0.5)
    translate_y = y_range * (np.random.rand() - 0.5)

    steering_angle += translate_x*0.002 #add a small amount of noise
    translation_matrix = np.float32([[1, 0, translate_x],[0, 1, translate_y]])
    height, width, depth = image.shape
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    return image, steering_angle


def change_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4*(np.random.rand()-0.5)
    hsv[:,:,2] = hsv[:,:,2]*ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def colour_space_transform(image):
    image = np.array(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def crop_image(image):
    return image[60:-25, :, :]

def resize_image(image):
    return sktrans.resize(image, training_image_size)


def add_shadow(image):
    h, w, d = image.shape

    [x1,x2] = np.random.choice(w,2, replace=False)
    y1 = 0
    y2 = h

    xm, ym = np.mgrid[0:h, 0:w]

    mask = np.zeros_like(image[:,:,1])
    mask[(ym - y1)*(x2 - x1) - (y2 - y1)*(xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def flip_image(image, steering_angle):
    if np.random.rand() < 0.5:
        cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle

def choose_camera(center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return mpimg.imread(left), steering_angle + steering_correction
    elif choice ==1:
        return mpimg.imread(right), steering_angle - steering_correction

    return mpimg.imread(center), steering_angle

def generator(samples, batch_size = 32):
    while 1:

        X = []
        y = []

        i =0
        for ix in np.random.permutation(len(samples)):

            #Pick random sample from the dataset
            sample = samples[ix]
            center_path = sample[0]
            left_path = sample[1]
            right_path = sample[2]
            steering_angle = float(sample[3])


            # Choose one of the cameras and correct the steering angle to match
            image, steering_angle = augment_data(center_path, left_path, right_path, steering_angle)

            #image = preprocess(image)


            X.append(image)
            y.append(steering_angle)
            i +=1
            if i == batch_size:
                break

        yield np.array(X), np.array(y)
