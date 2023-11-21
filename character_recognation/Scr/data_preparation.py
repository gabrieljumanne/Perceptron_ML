# code for data collection and preparation
import os
from PIL import Image
from image_utils import convert_to_grayscale


def load_dataset(dataset_path):
    """
    load the original data set from specified path


    :param dataset_path: (string) file path or url of the data set from raw data
    :return: The loaded data set of the image
    """
    dataset = []    # an empty list to hold the  data

    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)  # all the file path
        if os.path.isfile(filepath):
            if filename.endswith((".png", ".png", ".jpeg")):    # filtering of the file i need
                dataset.append(filepath)

    return dataset


# # specifying the relative path from the current directory
# input_path = "../../character_recognation/Data/Raw_data"
# output_path = "../../character_recognation/Data/Processed_data"
# # my_dataset = load_dataset(relative_path)
# # print(my_dataset)


def preprocess_image(input_path, output_path):
    """
    perform data preprocessing on the individual image

    :param input_path: (string) path to the input image
    :param output_path:( string) path to the output image the processed image
    :return: This function has no the output value is just for processing the image
    """

    # load the image
    image = Image.open(input_path)

    # adding the preprocessing steps (ADD MORE AS NEEDED)
    # Example :resize the image to the required size
    resized_image = image.resize((422, 500))

    # Example : convert the image to grayscale
    grayscale_image = convert_to_grayscale(resized_image)

    # save the preprocessed image
    grayscale_image.save(output_path)


def main():
    """
    this code generalize the whole idea of data preparation in our OCR
    :return:
    """
    # Define the path for the of the original and processed data path
    raw_data_path = "../../character_recognation/Data/Raw_data"
    processed_data_path = "../../character_recognation/Data/processed_data"

    # create processed data directory if not present / exist
    os.makedirs(processed_data_path, exist_ok=True)

    # load the dataset
    dataset = load_dataset(raw_data_path)
    print(dataset)

    # perform the preprocessing on each image in dataset
    for input_image_path in dataset:
        # Define the corresponding output path in a processed data directory
        output_image_path = os.path.join(processed_data_path, os.path.basename(input_image_path))

        # preprocess the image and save it in processed data directory
        preprocess_image(input_image_path, output_image_path)


# if __name__ == "__main__":
#     main()

main()