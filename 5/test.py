import cv2
import numpy as np

def region_growing_floodfill(image, seed_point, threshold):
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Создаем маску для floodFill (на 2 пикселя больше по каждому измерению)
    flood_flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    lo_diff, up_diff = threshold, threshold  # Нижняя и верхняя границы для заливки
    rect = cv2.floodFill(image.copy(), mask, seed_point, 255, lo_diff, up_diff, flood_flags)
    return mask[1:-1, 1:-1] * 255  # Убираем дополнительные границы маски и возвращаем бинарное изображение

# Загрузка изображения
image_path = input("Enter path -> ")
image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image_original is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Задание параметров
seed_point = (214, 210)
threshold = 5

# Применение алгоритма
segmented_image = region_growing_floodfill(image_original, seed_point, threshold)

# Отображение результатов
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()