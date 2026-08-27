from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=(784,), name="pixels"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_ann",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
