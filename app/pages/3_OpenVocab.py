import _pathsetup  # noqa: F401  (puts repo root on sys.path under `streamlit run`)

import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yoloworld
from app.components.visualization import draw_prediction

st.title("Open-Vocabulary Detection — YOLO-World")

COMMON = ["person", "car", "dog", "cat", "bottle", "chair", "laptop", "cell phone",
          "traffic light", "backpack"]

selected = st.multiselect("Common classes", COMMON, default=["person"])
extra = st.text_input("Extra classes (comma-separated)", "")
classes = selected + [c.strip() for c in extra.split(",") if c.strip()]

conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None and not classes:
    st.warning("Select or type at least one class to detect.")
elif file is not None:
    image = Image.open(file).convert("RGB")
    with st.spinner("Running YOLO-World..."):
        pred = get_yoloworld().predict(np.array(image), classes=classes, conf=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"classes": classes, "labels": pred.labels})
