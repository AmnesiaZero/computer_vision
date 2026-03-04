import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml

# --- Загрузка и подготовка данных (MNIST) ---
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)
X = X.values  # Преобразуем DataFrame в numpy array

# Уменьшим выборку для ускорения работы
X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=42)

# Нормализация пикселей
X = X / 255.0

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Функции для извлечения признаков ---
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        img_reshaped = img.reshape(28, 28).astype(np.uint8)
        _, descriptors = sift.detectAndCompute(img_reshaped, None)
        if descriptors is not None:
            features.append(np.mean(descriptors, axis=0))
        else:
            features.append(np.zeros(128))
    return np.array(features)

def extract_haar_features(images):
    features = []
    for img in images:
        img_reshaped = img.reshape(28, 28).astype(np.uint8)
        sobelx = cv2.Sobel(img_reshaped, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_reshaped, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.append(magnitude.flatten())
    return np.array(features)

# --- Извлечение признаков ---
print("Извлечение признаков SIFT...")
start_time = time.time()
X_train_sift = extract_sift_features(X_train)
X_test_sift = extract_sift_features(X_test)
print(f"Время извлечения SIFT: {time.time() - start_time:.2f} сек")

print("Извлечение признаков Хаара...")
start_time = time.time()
X_train_haar = extract_haar_features(X_train)
X_test_haar = extract_haar_features(X_test)
print(f"Время извлечения Хаара: {time.time() - start_time:.2f} сек")

# --- Обучение и оценка kNN ---
def train_and_evaluate(X_train, X_test, feature_type):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность ({feature_type}): {accuracy:.4f}")

print("\nОценка точности:")
train_and_evaluate(X_train, X_test, "Пиксельные признаки")
train_and_evaluate(X_train_sift, X_test_sift, "SIFT")
train_and_evaluate(X_train_haar, X_test_haar, "Хаара")

# --- Визуализация результатов ---
# Используем numpy array вместо pandas для индексации
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

sample_indices = np.random.choice(len(X_test), 3, replace=False)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    # Исходное изображение
    plt.subplot(3, 4, i*4 + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Тест: {y_test_np[idx]}")
    plt.axis('off')

    # Ближайшие соседи по пикселям
    knn_pixel = KNeighborsClassifier(n_neighbors=1)
    knn_pixel.fit(X_train, y_train_np)
    _, neighbor_indices = knn_pixel.kneighbors([X_test[idx]])
    plt.subplot(3, 4, i*4 + 2)
    plt.imshow(X_train[neighbor_indices[0][0]].reshape(28, 28), cmap='gray')
    plt.title(f"Пиксель: {y_train_np[neighbor_indices[0][0]]}")
    plt.axis('off')

    # Ближайшие соседи по SIFT
    knn_sift = KNeighborsClassifier(n_neighbors=1)
    knn_sift.fit(X_train_sift, y_train_np)
    _, neighbor_indices = knn_sift.kneighbors([X_test_sift[idx]])
    plt.subplot(3, 4, i*4 + 3)
    plt.imshow(X_train[neighbor_indices[0][0]].reshape(28, 28), cmap='gray')
    plt.title(f"SIFT: {y_train_np[neighbor_indices[0][0]]}")
    plt.axis('off')

    # Ближайшие соседи по Хаара
    knn_haar = KNeighborsClassifier(n_neighbors=1)
    knn_haar.fit(X_train_haar, y_train_np)
    _, neighbor_indices = knn_haar.kneighbors([X_test_haar[idx]])
    plt.subplot(3, 4, i*4 + 4)
    plt.imshow(X_train[neighbor_indices[0][0]].reshape(28, 28), cmap='gray')
    plt.title(f"Хаара: {y_train_np[neighbor_indices[0][0]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

'''
выводы: пиксельные для простых датасетов, сифт для сложных изображений с вариациями масштаба, хаар не подходит для мниста,но может выполнять другие задачи
'''