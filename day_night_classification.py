import numpy as np
import cv2
import helpers

winname = "CV"
cv2.namedWindow(winname)

# Image data directories
image_dir_training = "images/day_night_images/training/"
image_dir_test = "images/day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

# Standardise all training images
STANDARDISED_LIST = helpers.standardise(IMAGE_LIST)

#cv2.imshow(winname, masked_image)
#cv2.waitKey()
cv2.destroyAllWindows()
