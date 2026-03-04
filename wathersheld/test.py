import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path

def run_demo(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.7 * distance.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
    plt.subplot(2, 3, 2), plt.imshow(thresh, cmap="gray"), plt.title("Threshold")
    plt.subplot(2, 3, 3), plt.imshow(sure_bg, cmap="gray"), plt.title("Sure Background")
    plt.subplot(2, 3, 4), plt.imshow(distance, cmap="gray"), plt.title("Distance Transform")
    plt.subplot(2, 3, 5), plt.imshow(sure_fg, cmap="gray"), plt.title("Sure Foreground")
    plt.subplot(2, 3, 6), plt.imshow(markers, cmap="jet"), plt.title("Markers")
    plt.tight_layout()
    plt.show()


def main():
    default_path = Path(__file__).resolve().parent / "8.jpg"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path
    run_demo(image_path)


if __name__ == "__main__":
    main()