import cv2
import numpy as np

winname = "CV"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

image = cv2.imread('images/thumbs_up_down.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow(winname, image)
cv2.waitKey()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Create a binary thresholded image
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

cv2.imshow(winname, binary)
cv2.waitKey()

# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 3)

cv2.imshow(winname, contours_image)
cv2.waitKey()

for c in contours:
    # Find the bounding rectangle of a selected contour
    x, y, w, h = cv2.boundingRect(c)
    # Crop to this size
    cropped_image = image[y: y + h, x: x + w]
    # Fit an ellipse to a contour and extract the angle from that ellipse
    (x, y), (MA, ma), angle = cv2.fitEllipse(c)

cv2.destroyAllWindows()

