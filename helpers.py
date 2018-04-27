# Helper functions

import os
import glob  # library for loading images from a directory
import cv2
import numpy as np


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["day", "night"]

    # Iterate through each color folder
    for im_type in image_types:

        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            # Read in the image
            im = cv2.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if im is not None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list


# This function should take in an RGB image and return a new, standardised version
# 600 height x 1100 width image size (px x px)
def standardise_input(image):
    # Resize image and pre-process so that all "standard" images are the same size
    standard_im = cv2.resize(image, (1100, 600))

    return standard_im


# Examples:
# encode("day") should return: 1
# encode("night") should return: 0
def encode(label):
    numerical_val = 0
    if label is 'day':
        numerical_val = 1
    # else it is night and can stay 0

    return numerical_val


# using both functions above, standardise the input images and output labels
def standardise(image_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardise_input(image)

        # Create a numerical label
        binary_label = encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, binary_label))

    return standard_list


# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:, :, 2])
    imshape = rgb_image.shape
    area = imshape[0] * imshape[1]

    # Calculate the average brightness using the area of the image
    # and the sum calculated above
    avg = sum_brightness / area

    return avg
