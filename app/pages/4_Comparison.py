import _pathsetup  # noqa: F401  (puts repo root on sys.path under `streamlit run`)

import streamlit as st
from PIL import Image

from app.components.model_runner import load_model
from app.components.comparison import run_comparison

st.title("Comparison")
st.caption("Detectors run one at a time (one model in memory at a time) to fit Streamlit Cloud.")

# (display name, model key)
ALL = [("YOLO11n", "yolo"), ("RF-DETR-nano", "rfdetr"), ("YOLO-World", "yoloworld")]
NAME_TO_KEY = dict(ALL)

chosen = st.multiselect("Detectors (pick 2–3)", [n for n, _ in ALL],
                        default=["YOLO11n", "RF-DETR-nano"])
wc_classes = st.text_input("YOLO-World classes (comma-separated)", "person, car")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None and not (2 <= len(chosen) <= 3):
    st.warning("Pick between 2 and 3 detectors.")
elif file is not None:
    image = Image.open(file).convert("RGB")

    specs = [(name, NAME_TO_KEY[name]) for name in chosen]
    per_kwargs = {}
    if "YOLO-World" in chosen:
        per_kwargs["YOLO-World"] = {
            "classes": [c.strip() for c in wc_classes.split(",") if c.strip()]
        }

    with st.spinner("Running models one at a time..."):
        results = run_comparison(image, specs, load_model, per_model_kwargs=per_kwargs)

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
