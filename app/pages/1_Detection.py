import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr
from app.components.visualization import draw_prediction

st.title("Detection")

model_name = st.selectbox("Model", ["YOLO11n", "RF-DETR-nano"])
conf = st.slider("Confidence / threshold", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)
    with st.spinner(f"Running {model_name}..."):
        if model_name == "YOLO11n":
            pred = get_yolo().predict(arr, conf=conf)
        else:
            pred = get_rfdetr().predict(arr, threshold=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"labels": pred.labels, "scores": [round(s, 3) for s in pred.scores]})
