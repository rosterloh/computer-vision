import numpy as np
import cv2

winname = "CV"
cv2.namedWindow(winname)

# Read in the image
image = cv2.imread('images/water_balloons.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

cv2.imshow(winname, image)
cv2.waitKey()

# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

# Define our color selection criteria in HSV values
lower_hue = np.array([160, 0, 0])
upper_hue = np.array([180, 255, 255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180, 0, 100])
upper_pink = np.array([255, 255, 230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(image)
masked_image[mask_rgb == 0] = [0, 0, 0]

# Visualise the mask
cv2.imshow(winname, masked_image)
cv2.waitKey()

# Now try HSV!

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(image)
masked_image[mask_hsv == 0] = [0, 0, 0]

# Visualise the mask
cv2.imshow(winname, masked_image)
cv2.waitKey()
cv2.destroyAllWindows()
