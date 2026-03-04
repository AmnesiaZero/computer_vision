import cv2 

image = cv2.imread(r'C:\Users\Vovchik\Desktop\computerVision\practice\2\yellowBall.jpg') 

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

color_low = (25,60,160) 
color_high = (60,255,255) 

only_object = cv2.inRange(hsv_img, color_low, color_high) 

moments = cv2.moments(only_object, 1) 

x_moment = moments['m01'] 

y_moment = moments['m10'] 

area = moments['m00'] 
 
x = int(x_moment / area) 

y = int(y_moment / area) 

cv2.putText(image, "Yellow ball", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 

cv2.putText(image, "%d, %d" % (x,y), (x,y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 

cv2.imshow('found', image) 
cv2.waitKey(0) 

