import cv2
import numpy as np

def region_growing(image, seed_point, threshold):
    rows, cols = image.shape # Получаем размеры изображения
    segmented_image = np.zeros_like(image, dtype=np.uint8) # Создаем пустое бинарное изображение таких же размеров
    visited = np.zeros_like(image, dtype=bool) # Создаем массив для отметки посещенных пикселей (изначально все не посещены)
    seed_x, seed_y = seed_point # Распаковываем координаты начальной точки
    if not (0 <= seed_x < rows and 0 <= seed_y < cols): # Проверка на корректность начальной точки
        raise ValueError("Seed point is out of bounds.")
    queue = [(seed_x, seed_y)] # Инициализируем очередь с начальной точкой. Будем использовать очередь (FIFO) для обработки пикселей в порядке их обнаружения.
    visited[seed_x, seed_y] = True # Отмечаем начальную точку как посещенную
    segmented_image[seed_x, seed_y] = 255 # Добавляем начальную точку в сегментированную область
    while queue: # Цикл продолжается, пока очередь не пуста
        x, y = queue.pop(0) # Извлекаем координаты следующего пикселя из очереди
        for dx in [-1, 0, 1]: # Проходим по окрестности 3x3 (без центрального пикселя)
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: # Пропускаем центральный пиксель
                    continue
                nx, ny = x + dx, y + dy # Координаты соседнего пикселя
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]: # Проверка на выход за границы изображения и на то, был ли пиксель уже посещен
                    if abs(image[x, y] - image[nx, ny]) <= threshold: # Проверяем условие сходства - разница интенсивностей меньше порогового значения?
                        queue.append((nx, ny)) # Добавляем подобный пиксель в очередь
                        visited[nx, ny] = True # Отмечаем его как посещенный
                        segmented_image[nx, ny] = 255 # Добавляем его в сегментированную область
    return segmented_image


image_path = input("Enter path ->") 
image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image_original is None:
    print(f"Error: Could not load image from {image_path}")
    exit()
# Выбор начальной точки (x, y)
seed_point = (214, 210) 
# Пороговое значение
threshold = 220
# Сегментация
segmented_image = region_growing(image_original, seed_point, threshold)
# Отображение результатов
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#порог, критерии сходства и центральный пиксель сегментировать, разного типа изображения    