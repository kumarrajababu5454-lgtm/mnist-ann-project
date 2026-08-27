from __future__ import annotations

from tensorflow.keras.datasets import mnist

from .preprocessing import prepare_mnist_images


def load_preprocessed_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return (
        prepare_mnist_images(train_images),
        train_labels,
        prepare_mnist_images(test_images),
        test_labels,
    )
