# user interface for the ocr system

from perceptron import Perceptron
from image_preprocessing import preprocess_image
from data_preparation import load_dataset
from PIL import Image  # for image manipulation
import os

def load_image(image_path):
    """
    Load the image from the specified file path

    :param image_path: path to the input image
    :return:PIL-image : The loaded image.
    """
    return Image.open(image_path)

def preprocess_input_image(image):
    """
    Preprocess the input image using the provided utilities
    :param image:(PIL) image - Input image
    :return:-numpy.ndarray - Preprocessed image array.
    """
    # preprocess the image using the image preprocessing module
    input_path = "../../character_recognation/Data/Raw_data"
    output_path = "../../character_recognation/Data/processed_data"

    preprocessed_images = preprocess_image(input_path, output_path)

    return preprocessed_images

def dispaly_results(predictions):
    """
    Display ocr prediction results .

    :param predictions:(list) - list of ocr predictions (out put for the perceptron)
    :return: no any return value
    """
    for i, prediction in enumerate(predictions):
        print(f"Image {i+1} prediction:{prediction}")

def main():
    # initialize the perceptron model
    perceptron_model = Perceptron(learning_rate=0.01, epochs=100)
    



