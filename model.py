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
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, Lambda, ELU, PReLU, Cropping2D
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam

# def resize_image(image):
#     shape = image.shape
#     image = image[math.floor(shape[0]/4):shape[0]-13, 0:shape[1]]
#     ratio = 100.0 / shape[1]
#     dim = (100, int(shape[0] * ratio))
#     resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     return img_to_array(resized_image)

def load_images(driving_data, core_path='./data/'):
    resized_images = []
    correction = 0.2
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
            resized_images.append(image_copy)
        float_angle = float(steering_angle)
        steerings.append(float_angle)
        steerings.append(float_angle + correction)
        steerings.append(float_angle - correction)
    return resized_images, steerings

# Load the data
data = {}
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
file_path = './data/driving_log.csv'
driving_data = pd.read_csv(file_path, names=columns)[1:]

images, steering_angles = load_images(driving_data)
X_train = np.array(images)
y_train = np.array(steering_angles)

# X_train, X_val, y_train, y_val = train_test_split(images, steering_angles,  test_size=0.2, random_state=1)

print('Loaded Images!!')

def generateImages(x, y):
    # Flipping images
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(x, y):
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_streering_angle = float(steering_angle) * -1.0
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_streering_angle)
    yield augmented_images, augmented_steering_angles


# model
img_shape = (160, 320, 3)
print('Image Shape : ', img_shape)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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
model.add(Dropout(0.2))
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
#model.fit_generator(generateImages(X_train, y_train), samples_per_epoch=1000, nb_epoch=50, validation_data=generateImages(X_val, y_val))
model.fit(X_train, y_train, batch_size=100, nb_epoch=50,validation_split=0.2, shuffle=True)
# save model
print('Saving Model Weights!!')
model.save_weights('model.h5')
with open('model.json','w') as f:
    json.dump(model.to_json(),f,ensure_ascii=False)
