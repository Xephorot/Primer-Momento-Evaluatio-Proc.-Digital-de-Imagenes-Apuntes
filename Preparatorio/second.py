import numpy as np
import cv2
import matplotlib.pyplot as plt

def imprimir(imagen):
    plt.imshow(imagen, cmap="gray")
    plt.show()

image = cv2.imread('image.png')
I = image[:, :, (2, 1, 0)] 

# Imprimir imagen normal
imprimir(I)

# Imprimir imagen en escala de grises
Z = np.mean(I.astype(float), axis=2).astype(np.uint8)
imprimir(Z)

# Apply Gaussian Blur
blurred_Z = cv2.GaussianBlur(Z, (5, 5), 0)

# Apply Sobel edge detection
sobelX = cv2.Sobel(blurred_Z, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edge detection
sobelY = cv2.Sobel(blurred_Z, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edge detection
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Detect features using ORB
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(blurred_Z, None)

# Draw keypoints on the original image
keypoint_img = cv2.drawKeypoints(I, keypoints, None, color=(0,255,0), flags=0)

# Visualization
plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.title('Original Image')
plt.imshow(I)
plt.subplot(222)
plt.title('Blurred Grayscale Image')
plt.imshow(blurred_Z, cmap='gray')
plt.subplot(223)
plt.title('Edge Detection')
plt.imshow(sobelCombined, cmap='gray')
plt.subplot(224)
plt.title('Feature Detection')
plt.imshow(keypoint_img)
plt.show()
