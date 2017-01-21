import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import pandas as pd
import collections
import json
import math
import csv
import cv2
import os.path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, Lambda, ELU, PReLU
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam

def resize_image(image):
    shape = image.shape
    image = image[math.floor(shape[0]/4):shape[0]-13, 0:shape[1]]
    ratio = 100.0 / shape[1]
    dim = (100, int(shape[0] * ratio))
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return img_to_array(resized_image)

def load_resized_images(names, core_path='./data/'):
    resized_images = []
    for name in names:
        image = mpimg.imread(core_path + name)
        image_copy = np.copy(image)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        image_copy = resize_image(image_copy)
        resized_images.append(image_copy)
    return resized_images

# Load the data
data = {}
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

file_path = './data/driving_log.csv'
driving_data = pd.read_csv(file_path, names=columns)[1:]

# left_images = load_resized_images(driving_data['left'])
# right_images = load_resized_images(driving_data['right'])
center_images = np.array(load_resized_images(driving_data['center']))
steering_angles = np.array(driving_data['steering'])

print('Loaded Center Images!!')

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(center_images, steering_angles,  test_size=0.2, random_state=1)

# model
img_shape = (50, 100, 3)

print('Image Shape : ', img_shape)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape))
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(Activation('relu'))

model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Activation('relu'))

model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, batch_size=100, nb_epoch=50,validation_data=(X_val, y_val), shuffle=True)

# save model
print('Saving Model Weights!!')
model.save_weights('model.h5')
with open('model.json','w') as f:
    json.dump(model.to_json(),f,ensure_ascii=False)
