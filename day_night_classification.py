import random
import helpers


# This function should take in RGB image input
def estimate_label(rgb_image, threshold):
    # Extract average brightness feature from an RGB image
    avg = helpers.avg_brightness(rgb_image)

    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
    if avg > threshold:
        # if the average brightness is above the threshold value, we classify it as "day"
        predicted_label = 1
    # else, the predicted_label can stay 0 (it is predicted to be "night")

    return predicted_label


# Constructs a list of misclassified images given a list of test images and their labels
def get_misclassified_images(test_images, threshold=120):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im, threshold)

        # Compare true and predicted labels
        if predicted_label != true_label:
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Image data directories
image_dir_training = "images/day_night_images/training/"
image_dir_test = "images/day_night_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(image_dir_training)

# Standardise all training images
STANDARDISED_LIST = helpers.standardise(IMAGE_LIST)

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)

# Standardize the test data
STANDARDIZED_TEST_LIST = helpers.standardise(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))

for i in MISCLASSIFIED:
    print('Image with average brightness of ' + str(helpers.avg_brightness(i[0])) +
          ' misclassified as ' + ('day' if i[1] else 'night'))

best = num_correct
optimal = 120
for j in range(90, 130, 1):
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST, j)
    num_correct = total - len(MISCLASSIFIED)
    if num_correct > best:
        best = num_correct
        optimal = j

print('Optimal threshold value is ' + str(optimal))
