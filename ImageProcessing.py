'''
This file processes our spectrogram data. It crops the spectrograms as required and also divides them into many smaller images. 
'''

# Import statements.
import numpy as np
import cv2
import os
from random import shuffle
from PIL import Image, ImageChops
from tqdm import tqdm

# Variables that store the directory of respective spectrograms and location for sliced spectrograms.
classical_visual = 'Spectrograms/Classical'
jazz_visual = 'Spectrograms/Jazz'
metal_visual = 'Spectrograms/Metal'
pop_visual = 'Spectrograms/Pop'
classical_slice_visual = 'Spectrograms/ClassicalSlices'
jazz_slice_visual = 'Spectrograms/JazzSlices'
metal_slice_visual = 'Spectrograms/MetalSlices'
pop_slice_visual = 'Spectrograms/PopSlices'

# Takes in an image and eliminates unnecessary whitespacing as a result of saving images with matplotlib.
def trim(image):

    # Gets the background pixel color
    background = Image.new(image.mode, image.size, image.getpixel((0,0)))

    # Finds the difference between the rest of the image and the background color
    difference = ImageChops.difference(image, background)

    # Modifies the difference to be used by bbox
    difference = ImageChops.add(difference, difference, 2.0, -100)

    # Gets the region of whitespacing
    bbox = difference.getbbox()

    # If the region is non-zero, then the region is cropped from the image
    if bbox:
        return image.crop(bbox)

# Takes in a directory of images and applies the 'trim' function defined above to every image in the directory. 
def crop_images(directory, genre, num_files):

    # Loops through all the files in the directory
    for x in tqdm(range(num_files)):

        # Creates full directory of the image
        file = os.path.join(directory, genre + str(x) + '.png')

        # Opens the file using the PIL
        image = Image.open(file)

        # Trims the image as required using the previously defined 'trim' function
        image = trim(image)

        # Saves the file
        image.save(file)

# Takes in a directory with images, a target directory, the genre of the corresponding audio, and the number of pieces to
# split the image vertically into and saves the resulting images into the target directory. 
def slice_images(directory, target_directory, genre, num_pieces):

    # Creates a counter to easily name the files
    counter = 0

    # Loops through every image in the supplied directory
    for image_path in tqdm(os.listdir(directory)):

        # Creates the full path for the image
        full_image_path = os.path.join(directory, image_path)

        # Opens the image using the PIL
        image = Image.open(full_image_path)

        # Defines variables to store the image's width and height
        width, height = image.size

        # Loops through the image by the number of desired split pieces
        for x in range(num_pieces):

            # Defines a variable that divides the width into the number of image pieces
            section_width = width / float(num_pieces)

            # Crops the image to the required dimensions
            frame = image.crop((x * section_width, 0, (x + 1) * section_width, height))
            
            #Saves the image to the target directory
            frame.save(os.path.join(target_directory, genre + str(counter) + '.' + str(x + 1) + '.png'))
        
        # Increments the counter to easily name the file
        counter += 1

# Takes in a directory with images and renames the files so that they can be randomly split into training and testing sets.
def shuffle_images(directory, genre, num_files):

    # Defines a Numpy array from 0 to the number of files in the directory
    numbers = np.arange(0, num_files)

    # The array is shuffled randomly
    shuffle(numbers)

    # A counter is defined to easily name the files
    counter = 0

    # Loops through every image in the directory
    for image in tqdm(os.listdir(directory)):

        # Creates a path to the current image
        old_path = os.path.join(directory, image)

        # Creates a path to the newly renamed image
        new_path = os.path.join(directory, genre + str(numbers[counter]) + '.png')

        # Renames the files from the old path
        os.rename(old_path, new_path)

        # Increments the counter for easy file naming
        counter += 1

# Applies the 'crop_images' function to the required directories.
crop_images(classical_visual, 'classical', 100)
crop_images(jazz_visual, 'jazz', 100)
crop_images(metal_visual, 'metal', 100)
crop_images(pop_visual, 'pop', 100)

# Applies the 'slice_images' function to the required directories.
slice_images(classical_visual, classical_slice_visual,'classical', 8)
slice_images(jazz_visual, jazz_slice_visual,'jazz', 8)
slice_images(metal_visual, metal_slice_visual,'metal', 8)
slice_images(pop_visual, pop_slice_visual,'pop', 8)

# Applies the 'shuffle_images' function to the required directories.
shuffle_images(classical_slice_visual, 'classical', 800)
shuffle_images(jazz_slice_visual, 'jazz', 800)
shuffle_images(metal_slice_visual, 'metal', 800)
shuffle_images(pop_slice_visual, 'pop', 800)