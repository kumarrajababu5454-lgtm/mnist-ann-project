from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale MNIST pixel values from [0, 255] to [0, 1]."""
    return images.astype("float32") / 255.0


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten image tensors while preserving the batch dimension."""
    return images.reshape(images.shape[0], 28 * 28)


def prepare_mnist_images(images: np.ndarray) -> np.ndarray:
    return flatten_images(normalize_images(images))
