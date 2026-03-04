import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

# --- Загрузка данных Fashion-MNIST ---
print("Загрузка Fashion-MNIST...")
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = fashion_mnist.data, fashion_mnist.target.astype(int)

# Нормализация пикселей
X = X / 255.0

# Уменьшение выборки для ускорения (можно убрать для полного теста)
X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Обучение и оценка kNN в sklearn ---
print("\nОбучение kNN в sklearn...")
start_time = time.time()
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

start_time = time.time()
y_pred_sklearn = knn_sklearn.predict(X_test)
sklearn_pred_time = time.time() - start_time
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn kNN:")
print(f"- Точность: {accuracy_sklearn:.4f}")
print(f"- Время обучения: {sklearn_train_time:.2f} сек")
print(f"- Время предсказания: {sklearn_pred_time:.2f} сек")

# --- Обучение и оценка kNN в OpenCV ---
print("\nОбучение kNN в OpenCV...")
# Преобразование данных для OpenCV
X_train_cv = X_train.astype(np.float32)
y_train_cv = y_train.astype(np.float32)
X_test_cv = X_test.astype(np.float32)

start_time = time.time()
knn_opencv = cv2.ml.KNearest_create()
knn_opencv.train(X_train_cv, cv2.ml.ROW_SAMPLE, y_train_cv)
opencv_train_time = time.time() - start_time

start_time = time.time()
_, y_pred_opencv, _, _ = knn_opencv.findNearest(X_test_cv, k=3)
opencv_pred_time = time.time() - start_time
y_pred_opencv = y_pred_opencv.flatten().astype(int)
accuracy_opencv = accuracy_score(y_test, y_pred_opencv)

print(f"OpenCV kNN:")
print(f"- Точность: {accuracy_opencv:.4f}")
print(f"- Время обучения: {opencv_train_time:.2f} сек")
print(f"- Время предсказания: {opencv_pred_time:.2f} сек")

# --- Визуализация результатов ---
# Классы Fashion-MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Выборка для визуализации
sample_indices = np.random.choice(len(X_test), 5, replace=False)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(sample_indices):
    # Тестовое изображение
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    true_label = class_names[y_test[idx]]
    pred_label_sklearn = class_names[y_pred_sklearn[idx]]
    pred_label_opencv = class_names[y_pred_opencv[idx]]
    plt.title(f"True: {true_label}\nSklearn: {pred_label_sklearn}\nOpenCV: {pred_label_opencv}")
    plt.axis('off')

    # Ближайшие соседи из sklearn
    _, neighbor_indices = knn_sklearn.kneighbors([X_test[idx]])
    plt.subplot(2, 5, i + 6)
    neighbors = X_train[neighbor_indices[0]].reshape(-1, 28, 28)
    neighbors_image = np.hstack([img for img in neighbors])
    plt.imshow(neighbors_image, cmap='gray')
    plt.title(f"Ближайшие соседи (sklearn)")
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- Сравнительная таблица ---
print("\nСравнительная таблица:")
print("| Метрика       | sklearn | OpenCV |")
print("|---------------|---------|--------|")
print(f"| Точность      | {accuracy_sklearn:.4f} | {accuracy_opencv:.4f} |")
print(f"| Время обучения | {sklearn_train_time:.2f} сек | {opencv_train_time:.2f} сек |")
print(f"| Время предсказания | {sklearn_pred_time:.2f} сек | {opencv_pred_time:.2f} сек |")