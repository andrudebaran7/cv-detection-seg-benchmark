import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr, get_yoloworld
from app.components.comparison import run_comparison

st.title("Comparison")

ALL = ["YOLO11n", "RF-DETR-nano", "YOLO-World"]
chosen = st.multiselect("Detectors (pick 2–3)", ALL, default=["YOLO11n", "RF-DETR-nano"])
wc_classes = st.text_input("YOLO-World classes (comma-separated)", "person, car")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None and not (2 <= len(chosen) <= 3):
    st.warning("Pick between 2 and 3 detectors.")
elif file is not None:
    image = Image.open(file).convert("RGB")

    loaders = {"YOLO11n": get_yolo, "RF-DETR-nano": get_rfdetr, "YOLO-World": get_yoloworld}
    models = {name: loaders[name]() for name in chosen}
    per_kwargs = {}
    if "YOLO-World" in models:
        per_kwargs["YOLO-World"] = {
            "classes": [c.strip() for c in wc_classes.split(",") if c.strip()]
        }

    with st.spinner("Running models..."):
        results = run_comparison(image, models, per_model_kwargs=per_kwargs)

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        col.image(r.image, caption=r.name)
    st.table(
        {
            "model": [r.name for r in results],
            "objects": [len(r.prediction) for r in results],
            "latency_ms": [round(r.prediction.latency_ms) for r in results],
        }
    )
