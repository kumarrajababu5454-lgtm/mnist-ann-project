from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from .data_loader import load_preprocessed_mnist
from .predict import MODEL_PATH, load_model

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def evaluate(model_path: Path = MODEL_PATH) -> None:
    _, _, x_test, y_test = load_preprocessed_mnist()
    model = load_model(model_path)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print(f"test_loss: {loss:.4f}")
    print(f"test_accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, digits=4))

    ARTIFACTS.mkdir(exist_ok=True)
    matrix = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(matrix).plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    evaluate()
