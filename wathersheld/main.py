import cv2
import numpy as np
from matplotlib import pyplot as plt

def watershed_segmentation(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Серое изображение")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(sure_fg, cmap='gray')
    plt.title("Передний план")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(unknown, cmap='gray')
    plt.title("Неизвестная область")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Сегментированное изображение")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(markers, cmap='jet')
    plt.title("Маркеры")
    plt.axis("off")
    
    plt.show()


image_path = 'C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\wathersheld\\8.jpg'
watershed_segmentation(image_path)
