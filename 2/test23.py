import cv2
import numpy as np

image = cv2.imread(r'C:\Users\Vovchik\Desktop\computerVision\practice\2\yellowBall.png')

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_low = (25, 60, 100)
color_high = (60, 255, 255)

only_object = cv2.inRange(hsv_img, color_low, color_high)

kernel = np.ones((5, 5), np.uint8)
only_object = cv2.morphologyEx(only_object, cv2.MORPH_CLOSE, kernel)
only_object = cv2.morphologyEx(only_object, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(only_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    center_x = x + w // 2
    center_y = y + h // 2

    cv2.putText(image, "Yellow ball", (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, "%d, %d" % (center_x, center_y), (center_x, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
else:
    cv2.putText(image, "Yellow ball not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('found', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

