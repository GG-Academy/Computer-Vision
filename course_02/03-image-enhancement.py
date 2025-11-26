import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("images/03-satellite.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE Enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1), plt.title("Original"), plt.imshow(gray, cmap="gray"), plt.axis("off")
plt.subplot(1,2,2), plt.title("Enhanced (CLAHE)"), plt.imshow(enhanced, cmap="gray"), plt.axis("off")
plt.show()
