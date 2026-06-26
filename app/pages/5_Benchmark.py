import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr, get_yoloworld
from app.components.benchmark import measure_latency, build_benchmark_rows

st.title("Benchmark — latency vs published mAP")
st.caption(
    "Latency is measured on this machine for the bundled sample image; mAP values are "
    "published headline numbers (YOLO-World is LVIS zero-shot)."
)

ALL = ["YOLO11n", "RF-DETR-nano", "YOLO-World"]
chosen = st.multiselect("Detectors", ALL, default=ALL)

if st.button("Run benchmark"):
    if len(chosen) < 2:
        st.warning("Pick at least two detectors.")
    else:
        image = Image.open("data/sample_images/bus.jpg").convert("RGB")
        arr = np.array(image)
        loaders = {"YOLO11n": get_yolo, "RF-DETR-nano": get_rfdetr, "YOLO-World": get_yoloworld}

        measured = {}
        with st.spinner("Measuring latency..."):
            for name in chosen:
                model = loaders[name]()
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
