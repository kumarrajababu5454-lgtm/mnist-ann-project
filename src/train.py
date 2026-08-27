from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras

from .data_loader import load_preprocessed_mnist
from .model import build_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist_ann.keras"


def train(epochs: int = 10, model_path: Path = MODEL_PATH) -> dict[str, float]:
    x_train, y_train, x_test, y_test = load_preprocessed_mnist()
    model = build_model()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    metrics = {
        "training_accuracy": float(history.history["accuracy"][-1]),
        "validation_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"saved_model: {model_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MNIST ANN.")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(epochs=args.epochs)
