import numpy as np
from scipy import misc
import csv
import json
import os.path
import pickle

# Pickle data

# Lets start with reading the log file to pickle the data
print('Reading from data folder the driving log.')
driving_log = []
with open('data/driving_log.csv','r') as f:
    datareader = csv.reader(f,delimiter=',')
    for row in datareader:
        driving_log.append(row)

# use csv data to set X to the images and y to the steering angles
print('Reading features: images, and labels: steering angles.')
num_images = len(driving_log) # num of labels
print("Total Number of Images: ", num_images)
labels = np.zeros(num_images)

for i,row in enumerate(driving_log):
    print(driving_log[i][0])
    file_name = driving_log[i][0].split('\\')[-1]
    image = misc.imread('./data/IMG/' + file_name)
    if(i % 100) == 0:
        print('Total Images Read: ',i)
    if (i == 0):
        images_concatenated = image
    elif(i < num_images):
        images_concatenated = np.concatenate((images_concatenated,image), axis=0)
    else:
        break
    labels[i] = driving_log[i][3]

# Reshape based on the dimensions used at the comma.ai predicting steering angles
features = images_concatenated.reshape(-1, 160, 320, 3)

print('Images concatenated are : ', len(images_concatenated))
print('Image Original Shape : ', images_concatenated[0].shape)
print('Images reshaped : ', features[0].shape)
print('Storing Pickled Data: ')
training_data = {'features': features, 'labels': labels}
pickle.dump(training_data, open('data/driving_data.p','wb'))
print('You have successfully pickled your data!')
