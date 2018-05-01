import cv2
import numpy as np

winname = "CV"
cv2.namedWindow(winname)

# Read in the image
image = cv2.imread('images/curved_lane.jpg')

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the image copy
cv2.imshow(winname, gray)
cv2.waitKey()

# Show Fourier Transform
# Normalise
norm_image = gray/255.0
f = np.fft.fft2(norm_image)
fshift = np.fft.fftshift(f)
f_image = 20*np.log(np.abs(fshift))
cv2.imshow(winname, f_image)
cv2.waitKey()

# Create a Gaussian blurred image
gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

# High-pass filter

# 3x3 sobel filters for edge detection
sobel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

# Filter the original and blurred grayscale images using filter2D
filtered = cv2.filter2D(gray, -1, sobel_x)
filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_y)

cv2.imshow(winname, filtered)
cv2.waitKey()
cv2.imshow(winname, filtered_blurred)
cv2.waitKey()

# Create threshold that sets all the filtered pixels to white
# Above a certain threshold

retval, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)

cv2.imshow(winname, binary_image)
cv2.waitKey()

# Canny combines all of these
# 1. Filter out noise with Gaussian Blur
# 2. Finds strength and direction of edges using Sobel filters
# 3. Applies non-maximum suppression to isolate the strongest edges and thin them to 1 pixel wide lines
# 4. Uses hysteresis thresholding to isolate best edges
edges = cv2.Canny(gray, 120, 240)
cv2.imshow(winname, edges)
cv2.waitKey()

cv2.destroyAllWindows()
