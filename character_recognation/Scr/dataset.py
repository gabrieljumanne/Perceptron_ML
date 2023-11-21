# this file is for the training data preparation
import os
import numpy as np
from PIL import Image

def load_processed_data(input_dir):
    # prepare the list of training data set
    x_train = []
    y_train = []

    # i need to load the images processed
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if os.path.isfile(image_path):
            if filename.endswith(".png", ".png", ".jpeg"):

                # open the image using PIL
                image = Image.open(image_path)

                # convert PIL image to numpy array
                img_array = np.array(image) / 255.0

                # append the image array to x_train
                x_train.append(img_array)

                

