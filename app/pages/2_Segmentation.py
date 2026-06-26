import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo_seg, get_sam2
from app.components.visualization import draw_prediction

st.title("Segmentation — YOLO11n-seg / SAM 2")

mode = st.radio("Model", ["YOLO11n-seg (automatic)", "SAM 2 tiny (point prompt)"])
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)

    if mode.startswith("YOLO"):
        with st.spinner("Running YOLO11n-seg..."):
            pred = get_yolo_seg().predict(arr)
    else:
        st.caption("Click point is the image center in v0.1 (interactive click lands in v0.2).")
        h, w = arr.shape[:2]
        with st.spinner("Running SAM 2 tiny..."):
            pred = get_sam2().predict(arr, points=[[w // 2, h // 2]])

    st.image(draw_prediction(image, pred), caption=f"{pred.latency_ms:.0f} ms")
