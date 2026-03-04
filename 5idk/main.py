import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def run_watershed(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    result_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb[markers == -1] = [255, 0, 0]
    return result_rgb, thresh, sure_bg, dist_transform, sure_fg


def main():
    default_path = Path(__file__).resolve().parent / "img_watershed_input_7.jpg"
    custom_path = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom_path) if custom_path else default_path

    result_rgb, thresh, sure_bg, dist_transform, sure_fg = run_watershed(image_path)

    plt.figure(figsize=(12, 8))
    panels = [
        ("Original + watershed borders", result_rgb, None),
        ("Binarization (INV + OTSU)", thresh, "gray"),
        ("Sure background", sure_bg, "gray"),
        ("Distance transform", dist_transform, "gray"),
        ("Sure foreground", sure_fg, "gray"),
    ]
    for idx, (title, content, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 3, idx)
        plt.imshow(content, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


