import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Vovchik\Desktop\computerVision\practice\2\bird.jpeg')

# Уменьшаем изображение в 2 раза
resized_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
cv2.imshow('Original Image', resized_image)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_ranges = {
    "red": [(0, 120, 70), (10, 255, 255)],
    "blue": [(100, 150, 0), (140, 255, 255)],
    "green": [(40, 40, 40), (80, 255, 255)],
    "lightBlue": [(85, 50, 50), (125, 255, 255)],
    "yellow": [(25, 100, 175), (35, 255, 255)],
    "pink": [(140, 100, 100), (170, 255, 255)],
    "black": [(0, 0, 0), (180, 255, 30)],
    "white": [(0, 0, 200), (180, 20, 255)]
}

for color, (low, high) in color_ranges.items():
    mask = cv2.inRange(hsv_img, np.array(low), np.array(high))
    cv2.imshow(f'{color}_mask', mask)

# Обнаружение объекта жёлтого цвета
yellow_mask = cv2.inRange(hsv_img, np.array(color_ranges["yellow"][0]), np.array(color_ranges["yellow"][1]))
contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Отображаем центр масс
        cv2.putText(image, f'({cx}, {cy})', (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
cv2.imshow('Yellow Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
