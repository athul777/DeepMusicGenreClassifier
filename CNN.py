'''
This file defines our convolutional network and trains it on the training set and validates it with the validation set. 
'''

# Imports necessary libraries.
import numpy as np
from DataLabelling import create_labelled_data

# Imports tflearn and all functions required from tflearn.
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Defines required directories, image dimensions to which images should be resized. 
training_data_dir = 'Training'
validation_data_dir = 'Validation'
img_height = 62
img_width = 369

# Defines parameters for the convolutional neural network.
learning_rate = 0.0002
epochs = 50
kernel = 3
pool = 3
dropout_rate = 0.5

# Defines a name for the model.
model_name = 'DeepMusicGenreClassifier'

# Creates training and validation data.
train = create_labelled_data(training_data_dir, 'training.npy')
validation = create_labelled_data(validation_data_dir, 'validation.npy')

# Use below if data is already stored in the respective files:
# train = np.load('training.npy')
# validation = np.load('validation.npy')

# 1st Layer of CNN: Input Layer 
# Input dimensions 'img_height' and 'img_width' resized as required
cnn = input_data(shape = [None, img_height, img_width, 1], name='input')

# 2nd Layer of CNN: Convolutional Layer + Max Pooling (64)
# A (3 x 3) kernel with default stride of 1 with a 'relu' activation function is applied
# A (3 x 3) pooling region is selected
cnn = conv_2d(cnn, 64, kernel, activation='relu')
cnn = max_pool_2d(cnn, pool)

# 3rd Layer of CNN: Convolutional Layer + Max Pooling (128)
# A (3 x 3) kernel with default stride of 1 with a 'relu' activation function is applied
# A (3 x 3) pooling region is selected
cnn = conv_2d(cnn, 128, kernel, activation='relu')
cnn = max_pool_2d(cnn, pool)

# 4th Layer of CNN: Convolutional Layer + Max Pooling (256)
# A (3 x 3) kernel with default stride of 1 with a 'relu' activation function is applied
# A (3 x 3) pooling region is selected
cnn = conv_2d(cnn, 256, kernel, activation='relu')
cnn = max_pool_2d(cnn, pool)

# 5th Layer of CNN: Convolutional Layer + Max Pooling (128)
# A (3 x 3) kernel with default stride of 1 with a 'relu' activation function is applied
# A (3 x 3) pooling region is selected
cnn = conv_2d(cnn, 128, kernel, activation='relu')
cnn = max_pool_2d(cnn, pool)

# 6th Layer of CNN: Convolutional Layer + Max Pooling (64)
# A (3 x 3) kernel with default stride of 1 with a 'relu' activation function is applied
# A (3 x 3) pooling region is selected
cnn = conv_2d(cnn, 64, kernel, activation='relu')
cnn = max_pool_2d(cnn, pool)

# 7th Layer of CNN: Fully Connected Layer with Dropout
# This layer is fully connected to the previous layer
# A dropout with probability 0.5 is applied to reduce overfitting
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, dropout_rate)

# 8th Layer of CNN: Output Layer
# This layer is fully connected to the previous layer
# An output vector of the form [c, j, m, p] where c, j, m, p are the probabilities for each music genre is returned
cnn = fully_connected(cnn, 4, activation='softmax')
cnn = regression(cnn, optimizer='Adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

# Uses the DNN function with 'cnn' and writes model details to log to be read by Tensorboard.
model = tflearn.DNN(cnn, tensorboard_dir='log')

# If the model already exists in the directory, load it.
if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print(model_name + ' loaded.')

# Defines the training vector 'X' (image) and 'y' (label).
X = np.array([i[0] for i in train]).reshape(-1, img_height, img_width, 1)
y = [i[1] for i in train]

# Defines the validation vector 'val_X' (image) and 'val_y' (label).
val_X = np.array([i[0] for i in validation]).reshape(-1, img_height, img_width, 1)
val_y = [i[1] for i in validation]

# Fits the model using the TFLearn 'fit' function. 
model.fit(X, y, n_epoch = epochs, validation_set = (val_X, val_y), show_metric = True, run_id = model_name)

# Saves the model using 'model_name'.
model.save(model_name)