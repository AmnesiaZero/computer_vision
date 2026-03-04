import numpy as np
from tensorflow import keras


def main():
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    sample_idx = np.random.randint(0, len(x_test), size=5)
    print("Random MNIST samples:")
    for idx in sample_idx:
        print(f"index={idx}, label={y_test[idx]}")


if __name__ == "__main__":
    main()
