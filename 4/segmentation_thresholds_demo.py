from pathlib import Path

import cv2


def demo_thresholds(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, global_thr = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, otsu_thr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Gray", image)
    cv2.imshow("Global threshold", global_thr)
    cv2.imshow("Otsu threshold", otsu_thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    default_path = Path(__file__).resolve().parent / "img_threshold_input_4.jpg"
    custom = input(f"Image path (Enter = {default_path}): ").strip()
    image_path = Path(custom) if custom else default_path
    demo_thresholds(image_path)


if __name__ == "__main__":
    main()


