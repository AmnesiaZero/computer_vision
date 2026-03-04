import cv2
import numpy as np
from collections import deque
from pathlib import Path
import argparse


def region_growing(image, seed_point, threshold):
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    seen = np.zeros_like(image, dtype=bool)

    seed_x, seed_y = seed_point
    if not (0 <= seed_x < rows and 0 <= seed_y < cols):
        raise ValueError("Seed point is out of bounds.")

    queue = deque([(seed_x, seed_y)])
    seen[seed_x, seed_y] = True
    segmented[seed_x, seed_y] = 255

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while queue:
        x, y = queue.popleft()
        current = int(image[x, y])

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if seen[nx, ny]:
                continue
            seen[nx, ny] = True
            if abs(current - int(image[nx, ny])) <= threshold:
                segmented[nx, ny] = 255
                queue.append((nx, ny))

    return segmented


def main():
    default_path = Path(__file__).resolve().parent / "img_region_growing_input_5.png"
    parser = argparse.ArgumentParser(description="Region growing segmentation.")
    parser.add_argument("--image", type=str, default="", help="Path to grayscale image")
    parser.add_argument("--seed-x", type=int, default=-1, help="Seed X (row), -1 = center")
    parser.add_argument("--seed-y", type=int, default=-1, help="Seed Y (col), -1 = center")
    parser.add_argument("--threshold", type=int, default=25, help="Similarity threshold")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else default_path
    image_original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_original is None:
        print(f"Error: Could not load image from {image_path}")
        return

    rows, cols = image_original.shape
    seed_x = rows // 2 if args.seed_x < 0 else args.seed_x
    seed_y = cols // 2 if args.seed_y < 0 else args.seed_y
    seed_point = (seed_x, seed_y)
    threshold = args.threshold

    print(f"[INFO] image={image_path}")
    print(f"[INFO] seed={seed_point}, threshold={threshold}")

    segmented_image = region_growing(image_original, seed_point, threshold)
    white_ratio = float(np.count_nonzero(segmented_image)) / segmented_image.size
    print(f"[INFO] segmented white ratio: {white_ratio:.3f}")
    print("[INFO] Press any key in image window to exit.")

    cv2.imshow("Original Image", image_original)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()