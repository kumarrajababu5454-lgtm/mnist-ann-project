import numpy as np
from PIL import Image

from backend.preprocessing import preprocess_uploaded_image
from src.preprocessing import flatten_images, normalize_images


def test_normalize_and_flatten():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    assert normalize_images(images).dtype == np.float32
    assert flatten_images(images).shape == (2, 784)


def test_uploaded_image_shape_and_scale():
    image = Image.new("L", (60, 40), color=0)
    contents = __import__("io").BytesIO()
    image.save(contents, format="PNG")
    result = preprocess_uploaded_image(contents.getvalue())
    assert result.shape == (1, 784)
    assert result.min() >= 0
    assert result.max() <= 1
