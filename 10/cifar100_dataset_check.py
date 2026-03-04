from tensorflow import keras


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    print(f"Train images: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}, labels: {y_test.shape}")
    print(f"Classes: {len(set(y_train.flatten().tolist()))}")


if __name__ == "__main__":
    main()
