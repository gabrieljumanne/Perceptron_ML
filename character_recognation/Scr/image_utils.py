# This is for the image utilities
# basic / common image utilities for ocr is conversion to grayscale and normalization

from PIL import Image
import numpy as np

def convert_to_grayscale(image):
    """
    Convert the input image from the grayscale. (dark(0) - white(255)).

    :param image: input image.
    :return: grayscale image .
    """

    return image.convert("L")

def normalize_image(image_array):
    """
    Normalize the pixel values for the input image

    :param image_array:(numpy.ndarray) - Input image array
    :return:(numpy.ndarray) Normalized image array
    """

    normalized_image = image_array / 255.0

    return normalized_image



