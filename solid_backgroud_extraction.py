import numpy as np
import cv2

winname = "CV"
cv2.namedWindow(winname)

# Read in the image
image = cv2.imread('images/pizza_bluescreen.jpg')

# Print out the type of image data and its dimensions (height, width, and color)
print('This image is:', type(image), ' with dimensions:', image.shape)

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# Display the image copy
cv2.imshow(winname, image_copy)
cv2.waitKey()

# Define the color selection boundaries in RGB values
lower_blue = np.array([0, 0, 200])
upper_blue = np.array([200, 200, 255])

# Define the masked area
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

# Visualise the mask
cv2.imshow(winname, mask)
cv2.waitKey()

# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
cv2.imshow(winname, masked_image)
cv2.waitKey()

# Load in a background image, and convert it to RGB
background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop it to the right size (514x816)
crop_background = background_image[0:514, 0:816]

# Mask the cropped background so that the pizza area is blocked
crop_background[mask == 0] = [0, 0, 0]

# Display the background
cv2.imshow(winname, crop_background)
cv2.waitKey()

# Add the two images together to create a complete image!
complete_image = masked_image + crop_background

# Display the result
cv2.imshow(winname, complete_image)
cv2.waitKey()
cv2.destroyAllWindows()
