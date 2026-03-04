import cv2
import numpy as np 

# Загрузка изображения
img = cv2.imread('D:\\Unik\\computerVision\\practice\\3\\cat.jpg')
# Преобразование в оттенки серого. Детектор Хариса работает с интенсивностью, поэтому цветное изображение нужно преобразовать в черно-белое.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Применение детектора углов Harris
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.06)
# Расширение результата для лучшей визуализации
dst = cv2.dilate(dst, None)
# Пороговое значение для выделения углов
img[dst > 0.01 * dst.max()] = [0, 0, 255] # Красные круги
# Отображение результата
cv2.imshow('Углы Harris', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
