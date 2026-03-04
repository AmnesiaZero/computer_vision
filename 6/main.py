import numpy as np 
import cv2 

def preprocess_image(image, target_size=(500, 500), blur_ksize=(5, 5)):
    image_resized = cv2.resize(image, target_size)

    image_blurred = cv2.GaussianBlur(image_resized, blur_ksize, 0)

    image_lab = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2LAB)

    return image_lab

def kmeans_segmentation(image, k): 
    pixels = image.reshape(-1, 3).astype(np.float32) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    segmented_image = centers[labels.flatten()].reshape(image.shape) 
    return segmented_image 

def mean_shift_segmentation(image, spatial_radius, color_radius, max_level):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    segmented_image = cv2.pyrMeanShiftFiltering(image_lab, spatial_radius, color_radius, maxLevel=max_level)
    
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2BGR)
    return segmented_image


spatial_radius = 100
color_radius = 20
max_level = 2
k = 5

image = cv2.imread('C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\6\\phone.jpg')
#segmented_image = kmeans_segmentation(image, k).astype('uint8')
segmented_image1 = mean_shift_segmentation(image, spatial_radius, color_radius, max_level)

#cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Segmented Image', segmented_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()