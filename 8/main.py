import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# 1. Загрузка и предобработка данных MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Нормализация пикселей (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Создание массива меток классов
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# 2. Построение модели
model = keras.Sequential([
    Flatten(input_shape=(28, 28)), # Преобразование 2D изображения в 1D вектор
    Dense(128, activation='relu'), # Полносвязный слой с 128 нейронами и ReLU активацией
    Dense(10, activation='softmax') # Выходной слой с 10 нейронами (для 10 цифр) и softmax активацией
])

# 3. Компиляция модели
model.compile(
    optimizer='adam', # Оптимизатор Adam 
    loss='categorical_crossentropy', # Функция потерь для многоклассовой классификации
    metrics=['accuracy'] # Метрика для оценки качества - точность
)

# 4. Обучение модели
history = model.fit(
    x_train, y_train,
    epochs=20, # Количество эпох обучения
    batch_size=32, # Размер батча
    validation_split=0.1 # 10% данных для валидации (контроль переобучения)
)

# 5. Оценка модели
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Вывод результатов обучения
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
