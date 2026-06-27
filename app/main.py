import streamlit as st

st.set_page_config(page_title="CV Detection & Segmentation Benchmark", layout="wide")

st.title("CV Detection & Segmentation Benchmark")
st.markdown(
    """
    Interactive comparison of object detection and image segmentation models through a
    single unified interface (v0.3).

    - **Detection** — YOLO11n or RF-DETR-nano on an uploaded image.
    - **Segmentation** — YOLO11n-seg (automatic), SAM 2 tiny (point-prompted), and
      Mask2Former (panoptic).
    - **Open-Vocabulary** — YOLO-World with predefined COCO classes plus free text.
    - **Comparison** — run 2–3 detectors side-by-side with a metrics table.
    - **Benchmark** — measured on-device latency vs published COCO mAP.

    Use the sidebar to pick a task. Models are nano/tiny to fit Streamlit Cloud; the
    Comparison and Benchmark pages load several models at once and are heavier.
    """
)
