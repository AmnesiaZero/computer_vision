import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def watershed_segmentation(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]

    panels = [
        ("Gray image", gray, "gray"),
        ("Foreground", sure_fg, "gray"),
        ("Unknown area", unknown, "gray"),
        ("Segmented image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None),
        ("Markers", markers, "jet"),
    ]

    plt.figure(figsize=(15, 10))
    for idx, (title, content, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 3, idx)
        plt.imshow(content, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    default_path = Path(__file__).resolve().parent / "8.jpg"
    custom_path = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom_path) if custom_path else default_path
    watershed_segmentation(image_path)


if __name__ == "__main__":
    main()
