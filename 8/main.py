import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    network = keras.Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    network.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return network


def plot_history(history):
    if history is None:
        return
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP training/inference.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if saved model exists")
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model_path = Path(__file__).resolve().parent / "mnist_mlp.keras"
    history = None

    if model_path.exists() and not args.retrain:
        print(f"[INFO] Loading saved model: {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("[INFO] Training new model...")
        model = build_model()
        history = model.fit(
            x_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
        )
        model.save(model_path)
        print(f"[INFO] Model saved: {model_path}")

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    plot_history(history)


if __name__ == "__main__":
    main()
