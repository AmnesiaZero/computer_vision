import cv2
import numpy as np
from collections import deque


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
    image_path = input("Enter path -> ").strip()
    image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_original is None:
        print(f"Error: Could not load image from {image_path}")
        return

    seed_point = (214, 210)
    threshold = 220
    segmented_image = region_growing(image_original, seed_point, threshold)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()