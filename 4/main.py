from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def run_segmentation_demo(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, global_thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    adaptive_thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, otsu_thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, threshold1=80, threshold2=160)

    plt.figure(figsize=(12, 8))
    panels = [
        ("Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None),
        ("Gray", gray, "gray"),
        ("Global threshold", global_thr, "gray"),
        ("Adaptive threshold", adaptive_thr, "gray"),
        ("Otsu threshold", otsu_thr, "gray"),
        ("Edges (Canny)", edges, "gray"),
    ]
    for index, (title, frame, cmap) in enumerate(panels, start=1):
        plt.subplot(2, 3, index)
        plt.imshow(frame, cmap=cmap)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    default_path = Path(__file__).resolve().parent / "img_threshold_input_4.jpg"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path
    run_segmentation_demo(image_path)


if __name__ == "__main__":
    main()


