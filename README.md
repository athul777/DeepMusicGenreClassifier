Dataset - http://marsyasweb.appspot.com/download/data_sets/

List of Software Used:
1. Anaconda (Python 3.6) with Jupyter Notebook - https://www.anaconda.com/
2. Tensorflow with GPU Support - https://www.tensorflow.org/
3. TFLearn - http://tflearn.org/
4. Numpy + Scipy - https://www.anaconda.com/
5. OpenCV - http://docs.opencv.org/3.0-beta/index.html
6. PIL - http://www.pythonware.com/products/pil/
7. tqdm - https://pypi.python.org/pypi/tqdm

Overview of Python Files:
1. AudioProcessing.py - contains the code to process the raw audio and convert audio to visual data.
2. ImageProcessing.py - contains code to modify and optimize generated imaged to be fed to the network.
3. DataLabelling.py - contains functions that take data from the Training, Testing, and Validation folders and generates data that can be fed to the network with labels.
4. CNN.py - contains all functions to build the CNN and also trains the network.
5. TestModel.py - contains everything in CNN.py and also trains it on a testing set of images located in the 'Testing' directory.

Note: the running instructions below are only if you want to run everything yourself again. All the exact same results are available in the IPython notebook.

Running Instructions (Executing Python files from scratch with just audio files without a Jupyter Notebook):
Prerequisites: Download dataset, extract the Classical, Jazz, Metal, and Pop audio files, install all software mentioned above. Move dataset to Audio/{genre} where audio files are separated by genre after converting the audio to mono.
1. Make sure the Audio files are located in 'Audio/{genre}' , where genre is either Classical, Jazz, Metal, or Pop.
2. Run AudioProcessing.py to generate images.
3. Run ImageProcessing.py to process the images. Your images should all be in the 'Spectrograms/' directory.
4. From the folders titles '{genre}Slices', divide the dataset into training, testing, and validation sets (and place them in folders with respective titles in the main directory).
5. Run CNN.py to train the network (This takes a very long time without a discrete GPU).
6. Run TestModel.py to see how the network performs on the testing set.

Running Instruction (Just for the network):
1. If you wish to train the network more, please run CNN.py.
2. If you just wish to test the network on the testing set, please run TestModel.py