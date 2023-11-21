# code for the image preprocessing for OCR system

import os
from PIL import Image
import numpy as np
from image_utils import convert_to_grayscale, normalize_image
from data_preparation import load_dataset
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# images preprocessing techniques
def resize_image(image, target_size=(420, 500)):
    """
    Resize the input image to the desired size / target size

    :param image: (PIL image) input image for the resizing
    :param target_size:(tuple) target size for resizing the image
    :return: Resized image
    """

    return image.resize(target_size)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def preprocess_image(input_dir, output_dir):
    """
    Preprocess all the image from the input directory and save them in output directory

    :param input_dir: (str) Path to the input directory ( raw data)
    :param output_dir: (str) path to the output directory ( preprocess data)

    :return: array of processed dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = []     # for collecting the processed data

    for filename in os.listdir(input_dir):
        # filtering the desired files
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                output_path = os.path.join(output_dir, filename)

                # open the image using PIL
                original_image = Image.open(input_path)

                # Resize the image
                resized_image = resize_image(original_image)

                # convert the image to the grayscale
                grayscale_image = convert_to_grayscale(resized_image)

                # convert the PIL image to the numpy array
                image_array = np.array(grayscale_image)

                # Normalize the image array
                normalized_image = normalize_image(image_array)

                # save the processed array
                Image.fromarray((normalized_image * 255.0).astype(np.uint8)).save(output_path)
                # the save method returns None so they,  will leads to the value of the list to be None

                # append the output path
                dataset.append(output_path)
    return dataset

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# define the path for the input and output directory

input_directory = "../../character_recognation/Data/Raw_data"

output_directory = "../../character_recognation/Data/processed_data/"

# perform image preprocessing
hello = preprocess_image(input_directory, output_directory)
print(hello)





