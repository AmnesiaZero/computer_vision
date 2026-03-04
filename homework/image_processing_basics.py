import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarization(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(binary_thresh, cmap='gray'), plt.title('Threshold')
    plt.subplot(1, 3, 3), plt.imshow(adaptive_thresh, cmap='gray'), plt.title('Adaptive Threshold')
    plt.show()

def sobel_operator(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(np.abs(sobel_x), cmap='gray'), plt.title('Sobel X')
    plt.subplot(1, 3, 3), plt.imshow(np.abs(sobel_y), cmap='gray'), plt.title('Sobel Y')
    plt.show()

def canny_edge_detection(image_path, low_threshold, high_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray'), plt.title(f'Canny Edges ({low_threshold}, {high_threshold})')
    plt.show()

def compare_methods(image_path):
    print("Applying Sobel Operator:")
    sobel_operator(image_path)
    print("Applying Canny Edge Detection:")
    canny_edge_detection(image_path, 50, 150)

# Example usage:
# binarization('image.jpg')
# sobel_operator('image.jpg')
# canny_edge_detection('image.jpg', 50, 150)
# compare_methods('image.jpg')
