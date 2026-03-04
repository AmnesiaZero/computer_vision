import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\wathersheld\\7.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(image, markers)
image_rgb[markers == -1] = [255, 0, 0]

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title('Оригинал + границы водораздела')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Бинаризация (THRESH_BINARY_INV + OTSU)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sure_bg, cmap='gray')
plt.title('Уверенный фон')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dist_transform, cmap='gray')
plt.title('Distance Transform')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sure_fg, cmap='gray')
plt.title('Уверенный объект')
plt.axis('off')

plt.tight_layout()
plt.show()
