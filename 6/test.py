import numpy as np
import cv2
from pathlib import Path


def preprocess_image(image, target_size=(300, 300), blur_ksize=(5, 5)):
    resized = cv2.resize(image, target_size)
    blurred = cv2.GaussianBlur(resized, blur_ksize, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)


def kmeans_segmentation(image, k):
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers[labels.flatten()].reshape(image.shape).astype(np.uint8)


def main():
    default_path = Path(__file__).resolve().parent / "random_cluster_test_6.jpg"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    prepared = preprocess_image(image)
    segmented_lab = kmeans_segmentation(prepared, k=2)
    segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)

    cv2.imshow("Segmented Image (k-means)", segmented_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
