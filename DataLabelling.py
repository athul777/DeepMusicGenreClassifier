'''
This file takes the spectrogram data and has functions that generate labels given the data. 
'''

# Import statements.
import numpy as np
import cv2
import os
from random import shuffle
from PIL import Image, ImageChops
from tqdm import tqdm

# Takes in an image and returns the corresponding label for that image based on its name.
def create_label(image):

    # Defines the genre of the image using its name
    genre_raw = image.split('.')[0]

    # Removes all numbers from 'genre_raw'
    genre = ''.join([i for i in genre_raw if not i.isdigit()])

    # Defines which label to return based on the string 'genre'
    if genre == 'classical': return [1, 0, 0, 0]
    elif genre == 'jazz': return [0, 1, 0, 0]
    elif genre == 'metal': return [0, 0, 1, 0]
    elif genre == 'pop': return [0, 0, 0, 1]

# Takes in a directory and labels all the images in that directory and stores the data in a file.
def create_labelled_data(directory, file_name):

    # Defines empty array that will be filled with labelled data
    labelled_data = []

    # Loops through every image in the directory
    for image in os.listdir(directory):

        # Stores the label of the image
        label = create_label(image)

        # Creates full path to the image
        path = os.path.join(directory, image)

        # Reads the image as a grayscale image
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Resizes the image based on parameters we specified earlier
        image = cv2.resize(image, (image_height, image_width))

        # Appends both the image (as a numpy array) and its label to 'labelled_data'
        labelled.append([np.array(image), np.array(label)])
    
    # Randomly shuffles 'labelled_data'
    shuffle(labelled_data)

    # Saves 'labelled_data' to a file
    np.save(file_name, labelled_data)

    # Returns 'labelled_data'
    return labelled_data