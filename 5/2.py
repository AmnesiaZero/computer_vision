import cv2
import numpy as np

def split_and_merge(image, min_size, homogeneity_threshold):
    rows, cols = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    def process_region(x, y, width, height, region_image):
        """Рекурсивно обрабатывает регион изображения."""
        region_mean = np.mean(region_image)
        region_std = np.std(region_image)
        if width <= min_size or height <= min_size or region_std <= homogeneity_threshold:
            # Условие остановки: регион достаточно мал или однороден
            segmented_image[y:y+height, x:x+width] = region_mean # Заполняем регион средним значением
        else:
            # Разделение региона на 4 подрегиона
            half_width = width // 2
            half_height = height // 2
            process_region(x, y, half_width, half_height, region_image[:half_height, :half_width])          #Верхний левый
            process_region(x + half_width, y, half_width, half_height, region_image[:half_height, half_width:]) #Верхний правый
            process_region(x, y + half_height, half_width, half_height, region_image[half_height:, :half_width])#Нижний левый
            process_region(x + half_width, y + half_height, half_width, half_height, region_image[half_height:, half_width:]) #Нижний правый
    # Начало рекурсивного процесса со всего изображения
    process_region(0, 0, cols, rows, image)
    return segmented_image

# Пример использования:
image_path = input("Enter path ->") 
image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image_original is None:
    print(f"Error: Could not load image from {image_path}")
    exit()
min_region_size = 100 # Минимальный размер региона
homogeneity_threshold = 10 #Порог однородности
segmented_image = split_and_merge(image_original, min_region_size, homogeneity_threshold)
cv2.imshow("Original Image", image_original)
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
