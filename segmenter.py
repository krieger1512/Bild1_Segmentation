import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('path_to_your_image.jpg')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Apply Thresholding
_, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 5: Find Contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw Contours
# Create an empty image to draw contours
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Optionally, draw contours on the original image
contour_image_on_original = image.copy()
cv2.drawContours(contour_image_on_original, contours, -1, (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Contours on Blank Image')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Contours on Original Image')
plt.imshow(cv2.cvtColor(contour_image_on_original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
