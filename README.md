## Model Description
Going through the assignment I read the end to end learning paper from NVIDIA , and based on a lot of other
classmates recommendations I decided to stick with nvidia model to train. I have done some data visualization
as well to help me visualize the data and see what is my current steering angles distribution.

Based on great blog posts from Vivek, Annie and Nick I was able to visualize and realize that I can crop my image
and resize it which made my model much more efficient and faster.

I initially experimented with the comma.ai model, however looking at the paper and through experimentation my car
was able to drive longer before heading into the water through the NVIDIA model

For Pre processing, I have re-sized the images to make training faster, and I have also cropped the bottom of the image to make sure
the tip top of the car doesn't appear anymore

Here is the initial comma.ai model which I used:

,,,,

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

,,,,

You can see the model from comma ai has three Convolutional layers and uses ELU activation functions.

The NVIDIA model on the other hand:

,,,,

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

,,,,

The NVIDIA model on the other hand has six convulotionl layers, and uses relu activation functions.

Even though my model seems to have improved and my car does drive longer, I am still struggling with the car
getting into the water. From Vivek's and other team members posts it seems that preprocessing
the images might help