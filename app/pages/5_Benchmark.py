import _pathsetup  # noqa: F401  (puts repo root on sys.path under `streamlit run`)

import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import load_model
from app.components.benchmark import measure_latency, build_benchmark_rows

st.title("Benchmark — latency vs published mAP")
st.caption(
    "Latency is measured on this machine for the bundled sample image; mAP values are "
    "published headline numbers (YOLO-World is LVIS zero-shot). Detectors run one at a "
    "time to fit Streamlit Cloud memory."
)

# (display name, model key)
ALL = [("YOLO11n", "yolo"), ("RF-DETR-nano", "rfdetr"), ("YOLO-World", "yoloworld")]
NAME_TO_KEY = dict(ALL)

chosen = st.multiselect("Detectors", [n for n, _ in ALL], default=[n for n, _ in ALL])

if st.button("Run benchmark"):
    if len(chosen) < 2:
        st.warning("Pick at least two detectors.")
    else:
        image = Image.open("data/sample_images/bus.jpg").convert("RGB")
        arr = np.array(image)

        measured = {}
        with st.spinner("Measuring latency (one model at a time)..."):
            for name in chosen:
                model = load_model(NAME_TO_KEY[name])  # evicts the previous model
                if name == "YOLO-World":
                    measured[name] = measure_latency(model, arr, classes=["person", "car", "bus"])
                else:
                    measured[name] = measure_latency(model, arr)

        rows = build_benchmark_rows(measured)
        st.scatter_chart(
            {"latency_ms": [r["latency_ms"] for r in rows], "map": [r["map"] for r in rows]},
            x="latency_ms",
            y="map",
        )
        st.table(rows)
