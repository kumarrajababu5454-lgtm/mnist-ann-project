import numpy as np

from src.model import build_model


def test_model_shape_and_probabilities():
    model = build_model()
    assert model.input_shape == (None, 784)
    probabilities = model.predict(np.zeros((1, 784), dtype="float32"), verbose=0)[0]
    assert probabilities.shape == (10,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert 0 <= int(np.argmax(probabilities)) <= 9
