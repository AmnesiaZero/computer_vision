import cv2
import numpy as np


def split_and_merge(image, min_size, homogeneity_threshold):
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)

    def process_region(x, y, width, height):
        region = image[y:y + height, x:x + width]
        mean_value = float(np.mean(region))
        std_value = float(np.std(region))

        if width <= min_size or height <= min_size or std_value <= homogeneity_threshold:
            segmented[y:y + height, x:x + width] = int(mean_value)
            return

        half_w = max(1, width // 2)
        half_h = max(1, height // 2)
        right_w = width - half_w
        bottom_h = height - half_h

        process_region(x, y, half_w, half_h)
        process_region(x + half_w, y, right_w, half_h)
        process_region(x, y + half_h, half_w, bottom_h)
        process_region(x + half_w, y + half_h, right_w, bottom_h)

    process_region(0, 0, cols, rows)
    return segmented


def main():
    image_path = input("Enter path -> ").strip()
    image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_original is None:
        print(f"Error: Could not load image from {image_path}")
        return

    min_region_size = 100
    homogeneity_threshold = 10
    segmented_image = split_and_merge(image_original, min_region_size, homogeneity_threshold)

    cv2.imshow("Original Image", image_original)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
