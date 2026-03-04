import cv2
import numpy as np

def region_growing(image_path, seed_point, lo_diff=5, up_diff=5):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    fill_color = (0, 255, 0)

    flood_img = image.copy()
    _, _, _, rect = cv2.floodFill(
        flood_img, mask, seedPoint=seed_point, newVal=fill_color,
        loDiff=(lo_diff, lo_diff, lo_diff), upDiff=(up_diff, up_diff, up_diff)
    )

    visited_mask = cv2.inRange(flood_img, fill_color, fill_color)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Flood-Filled Image', flood_img)
    cv2.imshow('Visited Pixels', visited_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return flood_img, visited_mask


img_path = 'C:\\Users\\Vovchik\\Desktop\\computerVision\\practice\\idk\\unnamed.jpg'
seed = (142, 67) 
region_growing(img_path, seed)
