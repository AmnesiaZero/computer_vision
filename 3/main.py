from pathlib import Path

import cv2
import numpy as np


def detect_harris_corners(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.06)
    response = cv2.dilate(response, None)

    marked = image.copy()
    threshold = 0.01 * response.max()
    marked[response > threshold] = [0, 0, 255]
    return marked


def main():
    default_path = Path(__file__).resolve().parent / "cat.jpg"
    custom_path = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom_path) if custom_path else default_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    result = detect_harris_corners(image)
    cv2.imshow("Harris corners", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
