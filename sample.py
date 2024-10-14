import cv2
import numpy as np

# Read and resize the original image
image = cv2.imread('pexels-photo-1099680.jpeg')
image = cv2.resize(image, (300, 300))

# Convert to grayscale and resize
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (300, 300))

# Convert the grayscale image to 3 channels to match the original image
gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Apply Gaussian Blur and resize
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
blurred_image = cv2.resize(blurred_image, (300, 300))

# Convert the blurred image to 3 channels
blurred_image_3ch = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

# Apply Canny edge detection and resize
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
edges = cv2.resize(edges, (300, 300))

# Convert the edges image to 3 channels
edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


top_row = cv2.hconcat([image, gray_image_3ch])
bottom_row = cv2.hconcat([blurred_image_3ch, edges_3ch])
grid = cv2.vconcat([top_row, bottom_row])


cv2.imshow('Image Grid', grid)

cv2.waitKey(0)
cv2.destroyAllWindows()
