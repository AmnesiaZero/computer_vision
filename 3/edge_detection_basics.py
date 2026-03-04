import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_threshold(image_path, threshold_value=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def binarize_adaptive(image_path, block_size=11, C=2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    adaptive_binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, block_size, C)
    return adaptive_binary

def show_results(image_path):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary = binarize_threshold(image_path)
    adaptive_binary = binarize_adaptive(image_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Оригинальное изображение")
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Глобальная бинаризация")
    
    plt.subplot(1, 3, 3)
    plt.imshow(adaptive_binary, cmap='gray')
    plt.title("Адаптивная бинаризация")
    
    plt.show()

image_path = input("Enter path->")
show_results(image_path)