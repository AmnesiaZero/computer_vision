import cv2
import numpy as np
import matplotlib.pyplot as plt

img1_path = input("Enter path ->")
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) 
img2_path = input("Enter path ->")
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) 
img3_path = input("Enter path ->")
img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE) 
img4_path = input("Enter path ->")
img4 = cv2.imread(img4_path, cv2.IMREAD_GRAYSCALE) 

def alg1(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) 

    sobel = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 1.0, cv2.pow(sobely, 2.0), 1.0, 0.0)) 

    sobel_thresh = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)[1]
    return sobel

def alg2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

alg2(img1)
'''
plt.subplot(1, 2, 1)
plt.imshow(alg1(img1), cmap='gray')
plt.title("Sobel")

plt.subplot(1, 2, 2)
plt.imshow(alg1(img2), cmap='gray')
plt.title("Sobel")

plt.subplot(1, 2, 1)
plt.imshow(alg1(img3), cmap='gray')
plt.title("Sobel")

plt.subplot(1, 2, 2)
plt.imshow(alg1(img4), cmap='gray')
plt.title("Sobel")

plt.show()
'''
