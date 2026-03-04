from pathlib import Path
import argparse

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


def read_image_unicode_safe(path: Path):
    image = cv2.imread(str(path))
    if image is not None:
        return image
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except OSError:
        return None


def main():
    default_path = Path(__file__).resolve().parent / "img_cat_scene.jpg"
    parser = argparse.ArgumentParser(description="Detect Harris corners on an image.")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    args = parser.parse_args()
    image_path = Path(args.image) if args.image else default_path

    print(f"[INFO] Loading image: {image_path}")
    image = read_image_unicode_safe(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    result = detect_harris_corners(image)
    cv2.imshow("Harris corners", result)
    print("[INFO] OpenCV window is shown. Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

