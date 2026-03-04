from pathlib import Path
import argparse

import cv2
import numpy as np


def preprocess_image(image, target_size=(500, 500), blur_kernel=(5, 5)):
    resized = cv2.resize(image, target_size)
    blurred = cv2.GaussianBlur(resized, blur_kernel, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)


def kmeans_segmentation(image, clusters):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    segmented = centers[labels.flatten()].reshape(image.shape)
    return segmented.astype(np.uint8)


def mean_shift_segmentation(image, spatial_radius=100, color_radius=20, max_level=2):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    filtered = cv2.pyrMeanShiftFiltering(
        lab, spatial_radius, color_radius, maxLevel=max_level
    )
    return cv2.cvtColor(filtered, cv2.COLOR_Lab2BGR)


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
    default_path = Path(__file__).resolve().parent / "img_mean_shift_input_6.jpg"
    parser = argparse.ArgumentParser(description="Mean-shift image segmentation.")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument("--spatial", type=int, default=40, help="Spatial radius")
    parser.add_argument("--color", type=int, default=20, help="Color radius")
    parser.add_argument("--max-level", type=int, default=1, help="Pyramid max level")
    args = parser.parse_args()
    image_path = Path(args.image) if args.image else default_path

    print(f"[INFO] Loading image: {image_path}")
    image = read_image_unicode_safe(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(
        f"[INFO] Running mean-shift: spatial={args.spatial}, "
        f"color={args.color}, max_level={args.max_level}"
    )
    segmented = mean_shift_segmentation(
        image,
        spatial_radius=args.spatial,
        color_radius=args.color,
        max_level=args.max_level,
    )
    print("[INFO] Segmentation complete. Press any key in image window to exit.")
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Image", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

