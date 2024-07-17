import cv2
import numpy as np
from datetime import datetime

# Function to count objects and return timestamp
def count_objects(image_path):
    # Load an image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to enhance object boundaries
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to further enhance object features
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=2)

    # Perform object detection (using contour detection)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes of objects, filter based on contour area
    boxes = []
    min_contour_area = 200  # Adjust this threshold based on your image characteristics

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x+w, y+h))

    # Count objects
    object_count = len(boxes)

    return object_count

# Example usage
image_path = '2.jpg'
object_count = count_objects(image_path)
print(f"Total Number of Objects: {object_count}")
