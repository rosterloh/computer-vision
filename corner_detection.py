import cv2
import numpy as np

winname = "CV"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

image = cv2.imread('images/waffle.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Detect corners
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate corner image to enhance corner points
# Create a 3x3 kernel of ones
kernel = np.ones((3, 3), np.uint8)
dst = cv2.dilate(dst, kernel, iterations=1)

cv2.imshow(winname, dst)
cv2.waitKey()

# This value vary depending on the image and how many corners you want to detect
# Try changing this free parameter, 0.1, to be larger or smaller and see what happens
thresh = 0.1*dst.max()

# Create an image copy to draw corners on
corner_image = np.copy(image)

# Iterate through all the corners and draw them on the image (if they pass the threshold)
for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if dst[j, i] > thresh:
            # image, center pt, radius, color, thickness
            cv2.circle(corner_image, (i, j), 1, (0, 255, 0), 1)

cv2.imshow(winname, corner_image)
cv2.waitKey()
cv2.destroyAllWindows()
