import streamlit as st

from models.yolo_wrapper import YoloWrapper
from models.sam2_wrapper import Sam2Wrapper


@st.cache_resource
def get_yolo(weights: str = "yolo11n.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_yolo_seg(weights: str = "yolo11n-seg.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_sam2(weights: str = "sam2.1_t.pt") -> Sam2Wrapper:
    return Sam2Wrapper(weights)
