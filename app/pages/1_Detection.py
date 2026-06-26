import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo
from app.components.visualization import draw_prediction

st.title("Detection — YOLO11n")

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    with st.spinner("Running YOLO11n..."):
        pred = get_yolo().predict(np.array(image), conf=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"labels": pred.labels, "scores": [round(s, 3) for s in pred.scores]})
