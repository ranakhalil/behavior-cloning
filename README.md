**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model - I have utilized the nvidia model
* model_commai.py - Experimenting with other neural network architectures
* drive.py for driving the car in autonomous mode - added some preprocessing images code there
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results - You re reading it :)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json ( I am with the october cohort and my drive.py is a little older)
```

During training I have stored not only model.json , however also stored model files at different checking points during training
to make sure I have enough data in regards to the lifecycle training of my model and what are the best number of epochs to train on

The model checkpoints are provided as well with the github repo, along with files of older iterations that I hold onto to keep track
of how I tweaked my model. Initially I thought 15 epochs are were the magic is, however after few experimentations with 

#### 3. Submission code is usable and readable
Model.py:
    * Image preprocessing:
        
    During the image preprocessing phase I broke it down to three steps: 
        first step I read in all the images left, right and center and generated steering angle data for the right
        and left images
        second step I flipped the images and steering angle values to generate more data
        thrid step I resized the images since the top of the hood of the car at the bottom of the image is not necessary for training
        
    Here are the code snippets:

    ```
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
    ````

    Then after feedback I added a function for image brightness, and refactored flip_images:

    ````
    def change_brightness(image):
    change_pct = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness

    def preprocess_image_flip_and_brightness(x, y):
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(x, y):
        random_brightness = change_brightness(image)
        augmented_images.append(random_brightness)
        augmented_steering_angles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_streering_angle = float(steering_angle) * -1.0
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_streering_angle)
    return augmented_images, augmented_steering_angles

    ````

    ### Model Architecture and Training Strategy
Here is an overview summary of my final model:

````
Image Shape :  (50, 100, 3)
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
batchnormalization_1 (BatchNorma (None, 50, 100, 3)    200         batchnormalization_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 23, 48, 24)    1824        batchnormalization_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 10, 22, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 9, 48)      43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 7, 64)      27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      28736       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          75660       flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]
====================================================================================================
Total params: 321,087
Trainable params: 320,987
Non-trainable params: 100
____________________________________________________________________________________________________
````

#### 1. An appropriate model architecture has been employed

My model consists of five convolution neural network layers with [3x3, 5x5, 1x1, 1x7] filter sizes
in addition to five dense layers.

Convolutional neural networks are great for training on images, as the stanford image processing course site metioned:

````
 ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties
  into the architecture. These then make the forward function more efficient to implement and vastly reduce
  the amount of parameters in the network.

````

Convolutional neural networs are perfect as well to take an image with a lot of complex features and then distilling it
to simpler features, which is the perfect case for our problem here.

Here are my convultional layers:

model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,1,7,border_mode='valid', activation='relu', subsample=(1,1)))

The model includes RELU layers and a tanh layer to introduce nonlinearity, and the data is normalized in the model using a Keras BatchNormalization layer.

Here is the batch normalization layer before the convultional layers:
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=img_shape))

Here are example of the activation functions:

On the dense layers used relu and on the final layer used tanh:
````
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
````

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I have initially experimented with multiple dropout layers
like the old version of my model here:

````
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=img_shape))
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.2))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(ELU())
model.add(ELU())
model.add(Dense(1))
````
However I got really bad results with my car crashing early on, so I reduced the number of dropout layers to a single layer after the first fully connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually as follows:
````
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['mse', 'accuracy'])
````
I have also gathered metrics for accuracy and loss, and utilized mean square error to measure loss. I have also experimented with other optimizers, however stuck with the adam optimizer since it provided better results

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center , right and left images with generated steering angles. Also thanks to Annie Flippo, she provided me with some recovery data to be able to train my car to recover from crashing on the sides of the road recovering from the left and right sides of the road ...

i have also split my data into training and validation:
````
X_train, X_val, y_train, y_val = train_test_split(final_images, final_steering_angles,  test_size=0.2, random_state=1)
````
and made sure it is done randomly.

Due to the high volume of data compared to how much computing power I have, I have then used fit_generator to feed the model
the data in batches:

````

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
````

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to a. experiment with simpler models compared to the nvidia model like the comma.ai model and b. to get the best accuracy possible and teach my car how to recover successfully on its own.
After few iterations, I found that the end to end learning model from Nvidia is actually great, and works for the problem at hand.

My first step was to use a convolution neural network model similar to the Nvidia model, up until the last convultional layer I took the liberty of specifying a non symmetrical filter of 1x7 and got much better results rather than forcing the last convolutional layer to have a 1x1 filter
I thought this model might be appropriate because we are trying to solve a regression problem, and also because of how the nvidia end to end model have seemed to be designed for the problem of driving, and being done with experts it was a compelling starting point. I did however try to simplify the model itself by dropping some layers however I ended with some wacky results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I added some batchnormalization at the begining and experimented with dropouts and different activation functions

Then I attempted the number of epochs and added more data like the recovery data to give my model more information to train on

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track near or after the bridge to improve the driving behavior in these cases, I went back tweaked my model optimization parameters, convolutional layers and percentage of dropouts and when to do dropouts. I have also tweaked the activation functions


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with five convolutional layers and five dense layers:

````
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
````

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
````
Loaded Images!!
Image Shape :  (50, 100, 3)
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
batchnormalization_1 (BatchNorma (None, 50, 100, 3)    200         batchnormalization_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 23, 48, 24)    1824        batchnormalization_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 10, 22, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 9, 48)      43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 7, 64)      27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      28736       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64)            0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           6500        dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_4[0][0]
====================================================================================================
Total params: 135,427
Trainable params: 135,327
Non-trainable params: 100
____________________________________________________________________________________________________````
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. However due to difficulty in moving the car I have used the udacity data. Visualizations provided above the write up in this notebook with histograms and all :)

Then I worked on getting some recovery data, however since I had difficulty driving on my machine using the keyboard a student
graciously offered her recovery data

After the collection process, I had X number of data points. I then preprocessed this data by flipping the images, taking in the right and left images and generating steering angles.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25 as evidenced by storing each epoch's checkpoint to see what worked best. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I have used sklearn train_test_split to split the data into training and validation

#### 4. Visualizations

Please refer to [Visualization Notebook ](https://github.com/ranakhalil/behavior-cloning/blob/master/P3%20-%20Behavioral%20Cloning%20Visualization%20and%20PreProcessing%20for%20final%20writeup.ipynb)

#### 5. Model Accuracy:

````
Epoch 1/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0330 - mean_squared_error: 0.0330 - acc: 0.1817/home/Jake/anaconda3/lib/python3.5/site-packages/keras/engine/training.py:1573: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
22528/22519 [==============================] - 110s - loss: 0.0329 - mean_squared_error: 0.0329 - acc: 0.1812 - val_loss: 0.0287 - val_mean_squared_error: 0.0287 - val_acc: 0.1749
Epoch 2/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0274 - mean_squared_error: 0.0274 - acc: 0.1728Epoch 0022638/22519 [==============================] - 110s - loss: 0.0274 - mean_squared_error: 0.0274 - acc: 0.1730 - val_loss: 0.0265 - val_mean_squared_error: 0.0265 - val_acc: 0.1821
Epoch 3/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0231 - mean_squared_error: 0.0231 - acc: 0.1827Epoch 0022528/22519 [==============================] - 113s - loss: 0.0231 - mean_squared_error: 0.0231 - acc: 0.1825 - val_loss: 0.0226 - val_mean_squared_error: 0.0226 - val_acc: 0.1752
Epoch 4/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0238 - mean_squared_error: 0.0238 - acc: 0.1751Epoch 0022638/22519 [==============================] - 114s - loss: 0.0238 - mean_squared_error: 0.0238 - acc: 0.1756 - val_loss: 0.0232 - val_mean_squared_error: 0.0232 - val_acc: 0.1918
Epoch 5/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0205 - mean_squared_error: 0.0205 - acc: 0.1831Epoch 0022528/22519 [==============================] - 109s - loss: 0.0206 - mean_squared_error: 0.0206 - acc: 0.1831 - val_loss: 0.0202 - val_mean_squared_error: 0.0202 - val_acc: 0.1827
Epoch 6/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0217 - mean_squared_error: 0.0217 - acc: 0.1771Epoch 0022638/22519 [==============================] - 116s - loss: 0.0217 - mean_squared_error: 0.0217 - acc: 0.1768 - val_loss: 0.0207 - val_mean_squared_error: 0.0207 - val_acc: 0.1786
Epoch 7/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0186 - mean_squared_error: 0.0186 - acc: 0.1847Epoch 0022528/22519 [==============================] - 113s - loss: 0.0186 - mean_squared_error: 0.0186 - acc: 0.1847 - val_loss: 0.0200 - val_mean_squared_error: 0.0200 - val_acc: 0.1946
Epoch 8/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0198 - mean_squared_error: 0.0198 - acc: 0.1772Epoch 0022638/22519 [==============================] - 115s - loss: 0.0198 - mean_squared_error: 0.0198 - acc: 0.1770 - val_loss: 0.0192 - val_mean_squared_error: 0.0192 - val_acc: 0.1751
Epoch 9/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0170 - mean_squared_error: 0.0170 - acc: 0.1854Epoch 0022528/22519 [==============================] - 115s - loss: 0.0170 - mean_squared_error: 0.0170 - acc: 0.1852 - val_loss: 0.0194 - val_mean_squared_error: 0.0194 - val_acc: 0.1850
Epoch 10/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0181 - mean_squared_error: 0.0181 - acc: 0.1778Epoch 0022638/22519 [==============================] - 114s - loss: 0.0181 - mean_squared_error: 0.0181 - acc: 0.1779 - val_loss: 0.0192 - val_mean_squared_error: 0.0192 - val_acc: 0.1906
Epoch 11/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0157 - mean_squared_error: 0.0157 - acc: 0.1858Epoch 0022528/22519 [==============================] - 114s - loss: 0.0157 - mean_squared_error: 0.0157 - acc: 0.1863 - val_loss: 0.0181 - val_mean_squared_error: 0.0181 - val_acc: 0.1774
Epoch 12/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0168 - mean_squared_error: 0.0168 - acc: 0.1777Epoch 0022638/22519 [==============================] - 112s - loss: 0.0168 - mean_squared_error: 0.0168 - acc: 0.1778 - val_loss: 0.0186 - val_mean_squared_error: 0.0186 - val_acc: 0.1930
Epoch 13/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0144 - mean_squared_error: 0.0144 - acc: 0.1864Epoch 0022528/22519 [==============================] - 107s - loss: 0.0144 - mean_squared_error: 0.0144 - acc: 0.1862 - val_loss: 0.0168 - val_mean_squared_error: 0.0168 - val_acc: 0.1825
Epoch 14/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0156 - mean_squared_error: 0.0156 - acc: 0.1781Epoch 0022638/22519 [==============================] - 105s - loss: 0.0155 - mean_squared_error: 0.0155 - acc: 0.1785 - val_loss: 0.0174 - val_mean_squared_error: 0.0174 - val_acc: 0.1776
Epoch 15/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0136 - mean_squared_error: 0.0136 - acc: 0.1860Epoch 0022528/22519 [==============================] - 107s - loss: 0.0136 - mean_squared_error: 0.0136 - acc: 0.1860 - val_loss: 0.0167 - val_mean_squared_error: 0.0167 - val_acc: 0.1882
Epoch 16/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0147 - mean_squared_error: 0.0147 - acc: 0.1790Epoch 0022638/22519 [==============================] - 106s - loss: 0.0147 - mean_squared_error: 0.0147 - acc: 0.1790 - val_loss: 0.0170 - val_mean_squared_error: 0.0170 - val_acc: 0.1816
Epoch 17/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0129 - mean_squared_error: 0.0129 - acc: 0.1860Epoch 0022528/22519 [==============================] - 106s - loss: 0.0130 - mean_squared_error: 0.0130 - acc: 0.1858 - val_loss: 0.0165 - val_mean_squared_error: 0.0165 - val_acc: 0.1901
Epoch 18/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0138 - mean_squared_error: 0.0138 - acc: 0.1793Epoch 0022638/22519 [==============================] - 109s - loss: 0.0138 - mean_squared_error: 0.0138 - acc: 0.1800 - val_loss: 0.0171 - val_mean_squared_error: 0.0171 - val_acc: 0.1937
Epoch 19/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0121 - mean_squared_error: 0.0121 - acc: 0.1854Epoch 0022528/22519 [==============================] - 97s - loss: 0.0121 - mean_squared_error: 0.0121 - acc: 0.1851 - val_loss: 0.0157 - val_mean_squared_error: 0.0157 - val_acc: 0.1841
Epoch 20/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0130 - mean_squared_error: 0.0130 - acc: 0.1803Epoch 0022638/22519 [==============================] - 96s - loss: 0.0130 - mean_squared_error: 0.0130 - acc: 0.1800 - val_loss: 0.0166 - val_mean_squared_error: 0.0166 - val_acc: 0.1918
Epoch 21/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0118 - mean_squared_error: 0.0118 - acc: 0.1856Epoch 0022528/22519 [==============================] - 93s - loss: 0.0118 - mean_squared_error: 0.0118 - acc: 0.1857 - val_loss: 0.0159 - val_mean_squared_error: 0.0159 - val_acc: 0.1790
Epoch 22/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0123 - mean_squared_error: 0.0123 - acc: 0.1798Epoch 0022638/22519 [==============================] - 93s - loss: 0.0123 - mean_squared_error: 0.0123 - acc: 0.1798 - val_loss: 0.0153 - val_mean_squared_error: 0.0153 - val_acc: 0.1816
Epoch 23/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0111 - mean_squared_error: 0.0111 - acc: 0.1858Epoch 0022528/22519 [==============================] - 92s - loss: 0.0111 - mean_squared_error: 0.0111 - acc: 0.1854 - val_loss: 0.0163 - val_mean_squared_error: 0.0163 - val_acc: 0.1880
Epoch 24/25
22510/22519 [============================>.] - ETA: 0s - loss: 0.0115 - mean_squared_error: 0.0115 - acc: 0.1803Epoch 0022638/22519 [==============================] - 93s - loss: 0.0115 - mean_squared_error: 0.0115 - acc: 0.1803 - val_loss: 0.0160 - val_mean_squared_error: 0.0160 - val_acc: 0.1767
Epoch 25/25
22400/22519 [============================>.] - ETA: 0s - loss: 0.0105 - mean_squared_error: 0.0105 - acc: 0.1854Epoch 0022528/22519 [==============================] - 92s - loss: 0.0105 - mean_squared_error: 0.0105 - acc: 0.1855 - val_loss: 0.0160 - val_mean_squared_error: 0.0160 - val_acc: 0.1944
Saving Model Weights!!
````

As you can see the mean squared error is reducting and accuracy is increasing .. Great sign!!