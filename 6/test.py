import numpy as np
import cv2

def preprocess_image(image, target_size=(300, 300), blur_ksize=(5, 5)):
    image_resized = cv2.resize(image, target_size)

    image_blurred = cv2.GaussianBlur(image_resized, blur_ksize, 0)

    image_lab = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2LAB)

    return image_lab

def kmeans_segmentation(image, k):
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return segmented_image

image_path = 'C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\6\\stone.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

preprocessed_image = preprocess_image(image)

k = 2
segmented_image = kmeans_segmentation(preprocessed_image, k).astype('uint8')

segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)

cv2.imshow('Segmented Image', segmented_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
