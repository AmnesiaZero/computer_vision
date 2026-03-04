from pathlib import Path

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


def main():
    default_path = Path(__file__).resolve().parent / "phone.jpg"
    custom_path = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom_path) if custom_path else default_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Keeps behavior close to original: mean-shift result is shown.
    segmented = mean_shift_segmentation(image, spatial_radius=100, color_radius=20, max_level=2)
    cv2.imshow("Segmented Image", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()