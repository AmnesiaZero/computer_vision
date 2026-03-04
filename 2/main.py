from pathlib import Path
import argparse

import cv2
import numpy as np


COLOR_RANGES = {
    "red": ((0, 120, 70), (10, 255, 255)),
    "blue": ((100, 150, 0), (140, 255, 255)),
    "green": ((40, 40, 40), (80, 255, 255)),
    "lightBlue": ((85, 50, 50), (125, 255, 255)),
    "yellow": ((25, 100, 175), (35, 255, 255)),
    "pink": ((140, 100, 100), (170, 255, 255)),
    "black": ((0, 0, 0), (180, 255, 30)),
    "white": ((0, 0, 200), (180, 20, 255)),
}


def locate_image() -> Path:
    script_default = Path(__file__).resolve().parent / "img_bird_scene.jpeg"
    cwd_default = Path.cwd() / "img_bird_scene.jpeg"
    default_path = cwd_default if cwd_default.exists() else script_default
    parser = argparse.ArgumentParser(description="Detect colored objects in HSV.")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    args = parser.parse_args()
    return Path(args.image) if args.image else default_path


def read_image_unicode_safe(path: Path):
    """
    OpenCV on Windows can fail on non-ASCII paths with cv2.imread.
    Use fromfile + imdecode as a reliable fallback.
    """
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


def yellow_centers(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        yield center_x, center_y


def main():
    image_path = locate_image()
    print(f"[INFO] Loading image: {image_path}")
    image = read_image_unicode_safe(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    half_size = (image.shape[1] // 2, image.shape[0] // 2)
    preview = cv2.resize(image, half_size)
    cv2.imshow("Original Image", preview)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for color_name, (low, high) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv_image, np.array(low), np.array(high))
        cv2.imshow(f"{color_name}_mask", mask)

    yellow_low, yellow_high = COLOR_RANGES["yellow"]
    yellow_mask = cv2.inRange(hsv_image, np.array(yellow_low), np.array(yellow_high))
    output = image.copy()
    for center_x, center_y in yellow_centers(yellow_mask):
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(
            output,
            f"({center_x}, {center_y})",
            (center_x - 40, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Yellow Object Detection", output)
    print("[INFO] OpenCV windows are shown. Press any key in an image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

