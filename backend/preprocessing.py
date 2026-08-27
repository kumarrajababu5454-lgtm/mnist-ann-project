from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def preprocess_uploaded_image(contents: bytes) -> np.ndarray:
    """Convert a user image to a batched, flattened MNIST-like tensor."""
    try:
        image = Image.open(BytesIO(contents)).convert("L")
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype="float32")
    if pixels.mean() > 127:
        pixels = 255.0 - pixels
    return (pixels / 255.0).reshape(1, 784)
