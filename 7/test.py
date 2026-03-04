import numpy as np
import cv2
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        kp, desc = sift.detectAndCompute(img, None)
        if desc is None:
            desc = np.zeros((1, 128))
        features.append(np.mean(desc, axis=0))
    return np.array(features)

def extract_haar_features(images):
    features = []
    for img in images:
        haar_h = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        haar_v = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        feat = np.hstack([haar_h.mean(), haar_h.std(), haar_v.mean(), haar_v.std()])
        features.append(feat)
    return np.array(features)


def benchmark(name, x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    start = time.time()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    elapsed = time.time() - start
    acc = accuracy_score(y_test, predicted)
    print(f"sklearn kNN ({name}): accuracy={acc:.3f}, time={elapsed:.2f}s")


def benchmark_opencv(name, x_train, y_train, x_test, y_test):
    knn = cv2.ml.KNearest_create()
    start = time.time()
    knn.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
    _, predicted, _, _ = knn.findNearest(x_test.astype(np.float32), k=3)
    elapsed = time.time() - start
    acc = accuracy_score(y_test, predicted.ravel())
    print(f"OpenCV kNN ({name}): accuracy={acc:.3f}, time={elapsed:.2f}s")


def main():
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    X = X.reshape(-1, 28, 28).astype(np.uint8)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1000, test_size=1000, random_state=42, stratify=y
    )

    X_train_vec = X_train.reshape(len(X_train), -1)
    X_test_vec = X_test.reshape(len(X_test), -1)

    print("Extracting SIFT features...")
    X_train_sift = extract_sift_features(X_train)
    X_test_sift = extract_sift_features(X_test)

    print("Extracting Haar-like features...")
    X_train_haar = extract_haar_features(X_train)
    X_test_haar = extract_haar_features(X_test)

    benchmark("pixels", X_train_vec, y_train, X_test_vec, y_test)
    benchmark("SIFT", X_train_sift, y_train, X_test_sift, y_test)
    benchmark("Haar", X_train_haar, y_train, X_test_haar, y_test)

    benchmark_opencv("pixels", X_train_vec, y_train, X_test_vec, y_test)
    benchmark_opencv("SIFT", X_train_sift, y_train, X_test_sift, y_test)
    benchmark_opencv("Haar", X_train_haar, y_train, X_test_haar, y_test)


if __name__ == "__main__":
    main()
