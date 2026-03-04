import numpy as np
import cv2
from matplotlib import pyplot as plt

# Загрузка изображения
image = cv2.imread('C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\wathersheld\\8.jpg')
if image is None:
    raise ValueError("Изображение не загружено. Проверьте путь к файлу.")

# Преобразование в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки (Otsu's method)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Удаление шумов (морфологическое открытие)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Поиск области фона
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Поиск области переднего плана (используем преобразование расстояния)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Поиск неизвестной области
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Маркировка меток
ret, markers = cv2.connectedComponents(sure_fg)

# Добавляем 1 ко всем меткам, чтобы фон был не 0, а 1
markers = markers + 1

# Помечаем неизвестную область как 0
markers[unknown == 255] = 0

# Применяем алгоритм водораздела
markers = cv2.watershed(image, markers)

# Размечаем границы сегментов красным цветом
image[markers == -1] = [255, 0, 0]

# Отображение результатов
plt.figure(figsize=(12, 6))

plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(232), plt.imshow(thresh, cmap='gray'), plt.title('Threshold')
plt.subplot(233), plt.imshow(sure_bg, cmap='gray'), plt.title('Sure Background')
plt.subplot(234), plt.imshow(dist_transform, cmap='gray'), plt.title('Distance Transform')
plt.subplot(235), plt.imshow(sure_fg, cmap='gray'), plt.title('Sure Foreground')
plt.subplot(236), plt.imshow(markers, cmap='jet'), plt.title('Markers (Watershed)')

plt.tight_layout()
plt.show()