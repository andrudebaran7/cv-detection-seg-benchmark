import streamlit as st

st.set_page_config(page_title="CV Detection & Segmentation Benchmark", layout="wide")

st.title("CV Detection & Segmentation Benchmark")
st.markdown(
    """
    Interactive comparison of detection and segmentation models (v0.1).

    - **Detection** — YOLO11 nano on an uploaded image.
    - **Segmentation** — YOLO11-seg (automatic) and SAM 2 tiny (point-prompted).

    Use the sidebar to pick a task. Models are nano/tiny to fit Streamlit Cloud.
    """
)
