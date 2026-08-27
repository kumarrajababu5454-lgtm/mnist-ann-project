from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mnist_ann.keras"


def load_model(model_path: Path = MODEL_PATH) -> keras.Model:
    return keras.models.load_model(model_path)


def predict_image(model: keras.Model, image: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(model.predict(image, verbose=0)[0], dtype=float)
    predicted_digit = int(np.argmax(probabilities))
    return {
        "predicted_digit": predicted_digit,
        "confidence": float(probabilities[predicted_digit]),
        "probabilities": {
            str(digit): float(probabilities[digit]) for digit in range(10)
        },
    }
