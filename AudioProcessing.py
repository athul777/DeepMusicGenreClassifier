'''
This file contains the functions that will be used to create spectrograms from the audio obtained from GTZAN Genre Collection dataset.
'''

# Import statements.
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

# Variables that store the directory of respective audio files and desired locations of the spectrograms.
classical_audio = 'Audio/Classical'
classical_visual = 'Spectrograms/Classical'

jazz_audio = 'Audio/Jazz'
jazz_visual = 'Spectrograms/Jazz'

metal_audio = 'Audio/Metal'
metal_visual = 'Spectrograms/Metal'

pop_audio = 'Audio/Pop'
pop_visual = 'Spectrograms/Pop'

# Returns the sample rate and Numpy array of the audio.
def audio_info(audio_file):
    rate, data = wavfile.read(audio_file)
    return rate, data

# Creates a spectrogram given a (wav) audio file and saves it to the desired directory (input) with a name (input).
def create_spectrogram(audio_file, directory, name):

    # Executes audio_info defines above
    rate, data = audio_info(audio_file)

    # Length of windowing segments
    nfft = 256

    # Sampling Frequency
    fs = 256

    # Plots the spectrogram
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs)

    # Prevents axis from being displayed
    plt.axis('off')

    # Saves the name of the image (with its full directory)
    img_path = os.path.join(directory, name + '.png')

    # Saves the spectrogram to img_path
    plt.savefig(img_path, dpi=100, frameon='false', aspect='normal', bbox_inches='tight', pad_inches=0)

    plt.cla()

# Takes in a directory and runs create_spectrogram for every audio file in the directory.
def generate_spectrograms(audio_directory, picture_directory, genre, num_files):

    # Loops through all files in the directory
    for x in tqdm(range(num_files)):

        # If the file number is less than 10, then 4 zeros need to be appended to the name of the file
        if x < 10:
            file = os.path.join(audio_directory, genre + '.0000' + str(x) + '.wav')

        # Else only 3 need to be appended
        else:
            file = os.path.join(audio_directory, genre + '.000' + str(x) + '.wav')

        # Runs create_spectrogram for the file specified above
        create_spectrogram(file, picture_directory, genre + str(x))

# generate_spectrograms is run for every directory with audio.
generate_spectrograms(classical_audio, classical_visual, 'classical', 100)
generate_spectrograms(jazz_audio, jazz_visual, 'jazz', 100)
generate_spectrograms(metal_audio, metal_visual, 'metal', 100)
generate_spectrograms(pop_audio, pop_visual, 'pop', 100)