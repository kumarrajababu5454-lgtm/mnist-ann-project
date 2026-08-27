# MNIST ANN Handwritten Digit Classifier

A complete learning project that classifies handwritten digits with a fully connected Artificial Neural Network (ANN). A Streamlit frontend sends uploaded images over HTTP to a FastAPI backend, which preprocesses the image and returns the model's prediction and all ten class probabilities.

## Project Overview

MNIST contains grayscale images of handwritten digits from 0 through 9. This project intentionally uses an ANN rather than a CNN so the fundamentals of normalization, flattening, dense layers, training, evaluation, persistence, and serving remain visible. The model turns each 28 x 28 image into 784 input features because $28 x 28 = 784$.

## Architecture

```mermaid
flowchart TD
    User --> Streamlit[Streamlit frontend]
    Streamlit -->|POST /predict| FastAPI[FastAPI backend]
    FastAPI --> Preprocess[Grayscale, resize, normalize, flatten]
    Preprocess --> Model[Saved Keras ANN]
    Model --> FastAPI
    FastAPI -->|JSON prediction| Streamlit
```

## Machine Learning Workflow

Dataset -> preprocessing -> normalization -> flattening -> ANN -> training -> evaluation -> saved model -> prediction.

MNIST provides 60,000 training images and 10,000 test images. Each image is 28 x 28 pixels, grayscale, with one of 10 labels.

## Dataset

MNIST is downloaded automatically through TensorFlow/Keras when training or evaluation first runs. The dataset is not committed to this repository.

## Preprocessing

Training images are converted to `float32`, scaled from `[0, 255]` to `[0, 1]`, and flattened to 784 features. Uploaded images are converted to grayscale, resized to 28 x 28, inverted when necessary, normalized, flattened, and given a batch dimension of `(1, 784)`.

## Why ANN Instead of CNN?

This project uses an ANN to make normalization, flattening, dense layers, training, evaluation, persistence, and serving easy to study. CNNs generally preserve spatial relationships better and would normally be the stronger production choice for image classification.

## ANN Architecture

| Layer | Output | Activation |
|---|---:|---|
| Input | 784 | None |
| Dense | 128 | ReLU |
| Dense | 64 | ReLU |
| Output | 10 | Softmax |

## Technologies

Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn, Pillow, FastAPI, Uvicorn, Streamlit, Requests, and Pytest.

## Project Structure

- `src/`: dataset loading, preprocessing, model, training, evaluation, and prediction logic.
- `backend/`: FastAPI application, schemas, and uploaded-image preprocessing.
- `frontend/`: Streamlit application.
- `tests/`: preprocessing, model, and API tests.
- `models/`: saved `mnist_ann.keras` model.
- `artifacts/`: evaluation visualizations.
- `notebooks/`: beginner-friendly exploration notebook.
- `data/`: notes about automatic dataset downloads.

## Installation

```powershell
git clone <repository-url>
cd mnist-ann-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dataset downloads automatically through TensorFlow/Keras when training or evaluation first runs. No dataset or secret is required in the repository.

## Virtual Environment

Create and activate the project-local `.venv` before installing dependencies or running project commands.

## Training

```powershell
python -m src.train --epochs 10
```

This saves the model to `models/mnist_ann.keras` and prints training, validation, and test metrics.

## Evaluation

```powershell
python -m src.evaluate
```

The command prints a classification report and saves `artifacts/confusion_matrix.png`.

## Running Backend

```powershell
uvicorn backend.main:app --reload
```

For Render or another process-based host:

```text
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

- `GET /`: basic API message.
- `GET /health`: service health status.
- `POST /predict`: multipart image upload returning `predicted_digit`, `confidence`, and probabilities for digits 0 through 9.

## API Documentation

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`; ReDoc is at `/redoc`.

## Running Frontend

In a second terminal with the virtual environment activated:

```powershell
streamlit run frontend/streamlit_app.py
```

## Environment Variables

The frontend uses `BACKEND_URL`, defaulting to `http://127.0.0.1:8000`. Set it before launching when the backend is deployed:

```powershell
$env:BACKEND_URL = "https://your-backend.example.com"
streamlit run frontend/streamlit_app.py
```

The frontend and backend communicate through HTTP; the frontend does not import backend code.

## Testing

After training the model, run:

```powershell
pytest -q
```

Tests cover preprocessing shape and scale, model input/output behavior, API health, valid prediction uploads, and invalid file handling.

## Deployment

The backend and frontend are designed to deploy separately. Provide the deployed backend URL as `BACKEND_URL` to Streamlit Community Cloud. The documented Uvicorn command is suitable for Render-style services. Deployment is not claimed as complete by this repository.

## Live Demo

Frontend: To be deployed

Backend API: To be deployed

## Limitations

The ANN ignores much of the spatial structure in images, so a CNN would normally be a better production choice. Uploaded photos can also differ from the centered, standardized MNIST examples.

## Future Improvements

- Add a CNN comparison.
- Add optional drawing support.
- Improve image centering and digit cropping.
- Add Docker and CI/CD.
- Add model versioning and richer confidence visualizations.

## Learning Outcomes

This project demonstrates dataset handling, reusable preprocessing, dense neural-network design, model training, evaluation, saving/loading, API design, frontend/backend separation, automated testing, and deployment-oriented configuration.

## License

MIT
