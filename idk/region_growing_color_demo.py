import cv2
import numpy as np
from pathlib import Path

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


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent / "img_gui_segmentation_input.jpg"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path
    seed = (142, 67)
    region_growing(str(image_path), seed)


