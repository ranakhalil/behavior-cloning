## Model Description

Going through the assignment, initially I was planning on either:
1- building on top of the classification model build via keras in the lab
2- Create the nvidia network architecture provided in the paper
3- Work with built-in keras models to experiment with transfer learning

I ended up stumbling upon the following model from comma.ai:
https://github.com/commaai/research/blob/master/train_steering_model.py

I found this model easy to train on my machine and understand:
```
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
```

As show above there are three convolutional layers, two dropout layers and 
three exponential linear activation units. 

For improvements, I would like to increase the n_epoch and add two more
convolutional layers to make the model closer to Nvidia's network architecture
