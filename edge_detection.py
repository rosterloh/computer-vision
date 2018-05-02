import cv2
import numpy as np

winname = "CV"
cv2.namedWindow(winname)

# Read in the image
gray = cv2.imread('images/curved_lane.jpg', flags=cv2.IMREAD_GRAYSCALE)

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

# Hough Lines
image = cv2.imread('images/phone.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi / 180
threshold = 60
min_line_length = 100
max_line_gap = 5

line_image = np.copy(image)  # creating an image copy to draw lines on

# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the image copy
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

cv2.imshow(winname, line_image)
cv2.waitKey()

# Hough Circles
image = cv2.imread('images/round_farms.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# for drawing circles on
circles_im = np.copy(image)

# HoughCircles to detect circles
# right now there are too many, large circles being detected
# try changing the value of maxRadius, minRadius, and minDist
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1,
                           minDist=45,
                           param1=70,
                           param2=11,
                           minRadius=25,
                           maxRadius=30)

# convert circles into expected type
circles = np.uint16(np.around(circles))
# draw each one
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(circles_im, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(circles_im, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow(winname, circles_im)
cv2.waitKey()
cv2.destroyAllWindows()
