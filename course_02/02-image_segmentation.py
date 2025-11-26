import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("images/02-night.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = img_rgb.reshape((-1, 3))
pixels = np.float32(pixels)

# K-means settings
k = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Run K-means
_, labels, centers = cv2.kmeans(
    pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
segmented = centers[labels.flatten()].reshape(img_rgb.shape)

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img_rgb), plt.axis("off")
plt.subplot(1,2,2), plt.title("Segmented (K=3)"), plt.imshow(segmented), plt.axis("off")
plt.show()
