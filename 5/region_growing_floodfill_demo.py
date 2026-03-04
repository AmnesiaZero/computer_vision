import cv2
import numpy as np
from pathlib import Path


def region_growing_floodfill(image, seed_point, threshold):
    height, width = image.shape
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flood_flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(
        image.copy(),
        mask,
        seedPoint=seed_point,
        newVal=255,
        loDiff=threshold,
        upDiff=threshold,
        flags=flood_flags,
    )
    return mask[1:-1, 1:-1] * 255


def main():
    default_path = Path(__file__).resolve().parent / "img_region_growing_input_5.png"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path

    image_original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    segmented = region_growing_floodfill(image_original, seed_point=(214, 210), threshold=5)
    cv2.imshow("Segmented Image (FloodFill)", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

