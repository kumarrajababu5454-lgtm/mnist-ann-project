from __future__ import annotations

import os

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="MNIST ANN Classifier", page_icon="8", layout="centered")
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a digit image and let a small artificial neural network classify it.")

with st.expander("About the project"):
    st.write(
        "The model receives 28 x 28 grayscale pixels flattened into 784 values, "
        "then predicts one of the ten MNIST digit classes."
    )

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=220)
    if st.button("Classify digit", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"The backend could not be reached: {exc}")
        else:
            st.subheader("Prediction")
            st.metric("Predicted digit", result["predicted_digit"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.subheader("Class probabilities")
            st.bar_chart(result["probabilities"])

st.caption(f"Backend: {BACKEND_URL}")
