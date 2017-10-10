'''
This file is nearly identical to CNN.py, but instead of training the network, it loads it and performs predictions on the testing set. 
'''

# Imports necessary libraries
import numpy as np
from DataLabelling import create_labelled_data

# Imports tflearn and all functions required from tflearn.
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Defines testing set directory, image dimensions to which images should be resized. 
testing_data_dir = 'Testing'
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

# If the model already exists in the directory, load it, else terminate the program.
if os.path.exists('{}.meta'.format(model_name)):
    model.load(model_name)
    print(model_name + ' loaded.')
else:
    print('Please execute this file only after running CNN.py.')
    quit()

# Creates a testing set.
test = create_labelled_data(testing_data_dir, 'testing.npy')

# Defines the test vector 'X' (image) and 'y' (label).
X = np.array([i[0] for i in test]).reshape(-1, img_height, img_width, 1)
y = [i[1] for i in test]

# Get the predictions by inputting vector 'X' into the model.
y_hat = model.predict(X)

# Takes a list of predictions of the form n x (1 x 4) and returns a list of the form n x 1 where the 
# position of prediction (which corresponds to the genre) is returned.
def get_genres(predictions):

    # Sets 'genres' to an empty list
    genres = []

    # Gets the length of 'predictions'
    n = len(predictions)

    # Loops through the number of predictions
    for x in range(n):

        # Calculates the prediction by returning the index with the maximum value
        index = np.argmax(predictions[x])

        # Appends that index to 'genres'
        genres.append(index)
    
    # Returns 'genres'
    return genres

# Modifies 'y' and 'y_hat' to show only the predicted genres. 
y = get_genres(y)
y_hat = get_genres(y_hat)

# Returns a list that will have the ith index as 0 if the predictions match and non-zero otherwise.
error = np.subtract(y_hat, y)

# Calculates the accuracy of our model on the test set.
accuracy = (len(error) - np.count_nonzero(error)) / float(len(error))

# Prints the accuracy of the model on the test set. 
print(model_name + "'s accuracy on the test set is " + str(accuracy))