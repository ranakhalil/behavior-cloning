import numpy as np
import json
import pickle
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Lambda, Dropout, Lambda, ELU
from keras.optimizers import Adam

# Load the data
data = {}
with open('data/driving_data.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], random_state=0, test_size=0.33)

# normalize: mean zero and range -0.5 to 0.5
print('Normalizing data')
def normalize_data(x):
    x = x.astype('float32')
    x = x / 255 - 0.5
    return x
# will use Lambda instead
X_train = normalize_data(X_train)
X_val = normalize_data(X_val)

# model
img_shape = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=img_shape, output_shape=img_shape))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, batch_size=100, nb_epoch=100,validation_data=(X_val, y_val), shuffle=True)

# save model
print('Saving Model Weights!!')
model.save_weights('model.h5')
with open('model.json','w') as f:
    json.dump(model.to_json(),f,ensure_ascii=False)
