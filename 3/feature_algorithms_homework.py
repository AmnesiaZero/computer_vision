import cv2
import numpy as np

def orb(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp, des = orb.detectAndCompute(gray, None)

    img_orb = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('ORB Keypoints', img_orb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def akaze(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(gray, None)
    img_akaze = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('AKAZE Keypoints', img_akaze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def brisk(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brisk = cv2.BRISK_create()

    kp, des = brisk.detectAndCompute(gray, None)
    img_brisk = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('BRISK Keypoints', img_brisk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
while True:
    choice = input('1 for ORB, 2 for AKAZE, 3 for BRISK ->')
    image1_path = input("Enter path->")
    image2_path = input("Enter path->")

    if(choice == '1'):
        orb(image1_path)
        orb(image2_path)
    elif(choice == '2'):
        akaze(image1_path)
        akaze(image2_path)
    elif(choice == '3'):
        brisk(image1_path)
        brisk(image2_path)
    else:
        print('dam')
