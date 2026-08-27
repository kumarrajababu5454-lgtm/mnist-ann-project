import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_invalid_upload(client):
    response = client.post(
        "/predict", files={"file": ("note.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_image(client):
    image = Image.new("L", (28, 28), color=0)
    contents = io.BytesIO()
    image.save(contents, format="PNG")
    response = client.post(
        "/predict", files={"file": ("digit.png", contents.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"predicted_digit", "confidence", "probabilities"}
    assert 0 <= result["predicted_digit"] <= 9
    assert len(result["probabilities"]) == 10
