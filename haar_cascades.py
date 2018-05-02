import cv2
import numpy as np

winname = "CV"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

image = cv2.imread('images/multi_faces.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load in cascade classifier
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# run the detector on the grayscale image
faces = face_cascade.detectMultiScale(gray, 4, 6)

img_with_detections = np.copy(image)   # make a copy of the original image to plot rectangle detections ontop of

# loop over our detections and draw their corresponding boxes on top of our original image
for (x, y, w, h) in faces:
    # draw next detection as a red rectangle on top of the original image.
    # Note: the fourth element (255, 0, 0) determines the color of the rectangle,
    # and the final argument (here set to 5) determines the width of the drawn rectangle
    cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 5)

# display the result
cv2.imshow(winname, img_with_detections)
cv2.waitKey()
cv2.destroyAllWindows()
