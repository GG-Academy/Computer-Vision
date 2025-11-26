import cv2
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread("images/01-wing.jpg", cv2.IMREAD_GRAYSCALE)

# Canny edge detection
edges = cv2.Canny(img, threshold1=100, threshold2=200)

# Show results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(img, cmap="gray"), plt.axis("off")
plt.subplot(1,2,2), plt.title("Edges (Canny)"), plt.imshow(edges, cmap="gray"), plt.axis("off")
plt.show()
