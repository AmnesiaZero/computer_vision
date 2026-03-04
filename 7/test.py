import numpy as np
import cv2
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
X = X.reshape(-1, 28, 28).astype(np.uint8)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=1000, random_state=42, stratify=y)

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

X_train_vec = X_train.reshape(len(X_train), -1)
X_test_vec = X_test.reshape(len(X_test), -1)

print("Извлечение SIFT признаков...")
X_train_sift = extract_sift_features(X_train)
X_test_sift = extract_sift_features(X_test)

print("Извлечение Хаар-признаков...")
X_train_haar = extract_haar_features(X_train)
X_test_haar = extract_haar_features(X_test)
clf = KNeighborsClassifier(n_neighbors=3)
start = time.time()
clf.fit(X_train_vec, y_train)
y_pred_vec = clf.predict(X_test_vec)
sklearn_vec_time = time.time() - start
sklearn_vec_acc = accuracy_score(y_test, y_pred_vec)
print(f"sklearn kNN (pixels): accuracy={sklearn_vec_acc:.3f}, time={sklearn_vec_time:.2f}s")

clf = KNeighborsClassifier(n_neighbors=3)
start = time.time()
clf.fit(X_train_sift, y_train)
y_pred_sift = clf.predict(X_test_sift)
sklearn_sift_time = time.time() - start
sklearn_sift_acc = accuracy_score(y_test, y_pred_sift)
print(f"sklearn kNN (SIFT): accuracy={sklearn_sift_acc:.3f}, time={sklearn_sift_time:.2f}s")

clf = KNeighborsClassifier(n_neighbors=3)
start = time.time()
clf.fit(X_train_haar, y_train)
y_pred_haar = clf.predict(X_test_haar)
sklearn_haar_time = time.time() - start
sklearn_haar_acc = accuracy_score(y_test, y_pred_haar)
print(f"sklearn kNN (Haar): accuracy={sklearn_haar_acc:.3f}, time={sklearn_haar_time:.2f}s")

knn = cv2.ml.KNearest_create()
start = time.time()
knn.train(X_train_vec.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
_, y_pred_opencv, _, _ = knn.findNearest(X_test_vec.astype(np.float32), k=3)
opencv_vec_time = time.time() - start
opencv_vec_acc = accuracy_score(y_test, y_pred_opencv.ravel())
print(f"OpenCV kNN (pixels): accuracy={opencv_vec_acc:.3f}, time={opencv_vec_time:.2f}s")


knn = cv2.ml.KNearest_create()
start = time.time()
knn.train(X_train_sift.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
_, y_pred_opencv, _, _ = knn.findNearest(X_test_sift.astype(np.float32), k=3)
opencv_sift_time = time.time() - start
opencv_sift_acc = accuracy_score(y_test, y_pred_opencv.ravel())
print(f"OpenCV kNN (SIFT): accuracy={opencv_sift_acc:.3f}, time={opencv_sift_time:.2f}s")

knn = cv2.ml.KNearest_create()
start = time.time()
knn.train(X_train_haar.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))
_, y_pred_opencv, _, _ = knn.findNearest(X_test_haar.astype(np.float32), k=3)
opencv_haar_time = time.time() - start
opencv_haar_acc = accuracy_score(y_test, y_pred_opencv.ravel())
print(f"OpenCV kNN (Haar): accuracy={opencv_haar_acc:.3f}, time={opencv_haar_time:.2f}s")
