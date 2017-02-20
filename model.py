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
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, Lambda, ELU, PReLU, Cropping2D
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

def resize_image(image):
    shape = image.shape
    image = image[math.floor(shape[0]/4):shape[0]-13, 0:shape[1]]
    ratio = 100.0 / shape[1]
    dim = (100, int(shape[0] * ratio))
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return img_to_array(resized_image)

def load_images(driving_data, core_path='./data/'):
    resized_images = []
    correction = 0.05
    steerings = []
    for left_image, center_image, right_image, steering_angle in zip(driving_data['left'], driving_data['center'], driving_data['right'], driving_data['steering']):
        lname = core_path + left_image
        cname = core_path + center_image
        rname = core_path + right_image

        limage = mpimg.imread(lname.replace(" ", ""))
        cimage = mpimg.imread(cname.replace(" ", ""))
        rimage = mpimg.imread(rname.replace(" ", ""))

        for image in np.array([cimage, limage, rimage]):
            image_copy = np.copy(image)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            image_copy = resize_image(image_copy)
            resized_images.append(image_copy)
        float_angle = float(steering_angle)
        steerings.append(float_angle)
        steerings.append(float_angle + correction)
        steerings.append(float_angle - correction)
    return resized_images, steerings

def flip_values(x, y):
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(x, y):
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_streering_angle = float(steering_angle) * -1.0
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_streering_angle)
    return augmented_images, augmented_steering_angles


def generator(images, steering_angles, batch_size=32):
    num_samples = len(images)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset + batch_size]
            batch_steering_angles = steering_angles[offset:offset + batch_size]
            # trim image to only see section with road
            x = np.array(batch_images)
            y = np.array(batch_steering_angles)
            yield shuffle(x, y, random_state=0)


# Load the data
data = {}
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
file_path = './data/driving_log.csv'
driving_data = pd.read_csv(file_path, names=columns)[1:]

images, steering_angles = load_images(driving_data)
final_images, final_steering_angles = flip_values(images, steering_angles)

X_train, X_val, y_train, y_val = train_test_split(final_images, final_steering_angles,  test_size=0.2, random_state=1)


train_generator = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_val, y_val, batch_size=64)

print('Loaded Images!!')

# model
img_shape = X_train[0].shape

print('Image Shape : ', img_shape)

model = Sequential()
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=img_shape))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,1,7,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['mse', 'accuracy'])

model.summary()

# checkpoint
checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')

# fit the model
model.fit_generator(train_generator, samples_per_epoch=len(X_train), nb_epoch=25, validation_data=validation_generator, nb_val_samples=len(X_val), callbacks=[checkpoint])

# save model
print('Saving Model Weights!!')
model.save_weights('model.h5')
with open('model.json','w') as f:
    json.dump(model.to_json(),f,ensure_ascii=False)
