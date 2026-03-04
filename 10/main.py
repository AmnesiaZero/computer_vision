import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import argparse


def vgg_like(input_shape, classes):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(classes, activation="softmax"),
        ],
        name="VGG_like",
    )
    return model


def residual_block(x, filters):
    skip = x
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    if skip.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, padding="same")(skip)
    x = layers.Add()([x, skip])
    return layers.Activation("relu")(x)


def resnet_like(input_shape, classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="ResNet_like")


def dense_block(x, growth_rate, repeats):
    for _ in range(repeats):
        y = layers.BatchNormalization()(x)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(growth_rate, 3, padding="same")(y)
        x = layers.Concatenate()([x, y])
    return x


def densenet_like(input_shape, classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = dense_block(x, growth_rate=16, repeats=3)
    x = layers.MaxPooling2D()(x)
    x = dense_block(x, growth_rate=16, repeats=3)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="DenseNet_like")


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_eval(model, x_train, y_train, x_test, y_test, epochs=2):
    model = compile_model(model)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    params = model.count_params()
    return {"name": model.name, "accuracy": acc, "loss": loss, "params": params}


def eval_only(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    params = model.count_params()
    return {"name": model.name, "accuracy": acc, "loss": loss, "params": params}


def main():
    parser = argparse.ArgumentParser(description="Compare VGG-like, ResNet-like, DenseNet-like on CIFAR-100.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if saved models exist")
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Small subset keeps the comparison runnable on a regular PC.
    train_size = 10000
    test_size = 2000
    x_train, y_train = x_train[:train_size], y_train[:train_size]
    x_test, y_test = x_test[:test_size], y_test[:test_size]

    input_shape = (32, 32, 3)
    classes = 100
    model_builders = [
        ("VGG_like", lambda: vgg_like(input_shape, classes)),
        ("ResNet_like", lambda: resnet_like(input_shape, classes)),
        ("DenseNet_like", lambda: densenet_like(input_shape, classes)),
    ]
    model_dir = Path(__file__).resolve().parent / "saved_models"
    model_dir.mkdir(exist_ok=True)

    results = []
    for model_name, builder in model_builders:
        model_path = model_dir / f"{model_name}.keras"

        if model_path.exists() and not args.retrain:
            print(f"\n[INFO] Loading saved model: {model_path}")
            model = keras.models.load_model(model_path)
            results.append(eval_only(model, x_test, y_test))
        else:
            print(f"\nTraining {model_name} ...")
            model = builder()
            results.append(train_and_eval(model, x_train, y_train, x_test, y_test, epochs=2))
            model.save(model_path)
            print(f"[INFO] Model saved: {model_path}")

    print("\nComparison results:")
    for row in results:
        print(
            f"{row['name']}: acc={row['accuracy']:.4f}, "
            f"loss={row['loss']:.4f}, params={row['params']}"
        )


if __name__ == "__main__":
    main()
