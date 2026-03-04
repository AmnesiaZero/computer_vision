import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def make_gaussian_class(center_x, center_y, size=50, spread=1.0):
    x = np.random.normal(loc=center_x, scale=spread, size=size)
    y = np.random.normal(loc=center_y, scale=spread, size=size)
    return np.column_stack((x, y))


def main():
    np.random.seed(42)
    class_a = make_gaussian_class(2, 2)
    class_b = make_gaussian_class(7, 7)
    class_c = make_gaussian_class(4, 7)

    X = np.vstack((class_a, class_b, class_c))
    y = np.hstack(
        (
            np.zeros(len(class_a), dtype=int),
            np.ones(len(class_b), dtype=int),
            np.full(len(class_c), 2, dtype=int),
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(class_a[:, 0], class_a[:, 1], label="Class 1", color="red")
    plt.scatter(class_b[:, 0], class_b[:, 1], label="Class 2", color="green")
    plt.scatter(class_c[:, 0], class_c[:, 1], label="Class 3", color="blue")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker="x",
        color="black",
        label="Test points",
        s=100,
    )

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    mesh_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    mesh_labels = model.predict(mesh_points).reshape(grid_x.shape)
    plt.contourf(grid_x, grid_y, mesh_labels, alpha=0.2)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("k-NN point classification")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

