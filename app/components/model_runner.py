import streamlit as st

from models.yolo_wrapper import YoloWrapper
from models.sam2_wrapper import Sam2Wrapper
from models.rfdetr_wrapper import RfDetrWrapper
from models.yoloworld_wrapper import YoloWorldWrapper
from models.mask2former_wrapper import Mask2FormerWrapper


@st.cache_resource
def get_yolo(weights: str = "yolo11n.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_yolo_seg(weights: str = "yolo11n-seg.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_sam2(weights: str = "sam2.1_t.pt") -> Sam2Wrapper:
    return Sam2Wrapper(weights)


@st.cache_resource
def get_rfdetr(model_name: str = "nano") -> RfDetrWrapper:
    return RfDetrWrapper(model_name)


@st.cache_resource
def get_yoloworld(weights: str = "yolov8s-world.pt") -> YoloWorldWrapper:
    return YoloWorldWrapper(weights)


@st.cache_resource
def get_mask2former(model_id: str = "facebook/mask2former-swin-tiny-coco-panoptic") -> Mask2FormerWrapper:
    return Mask2FormerWrapper(model_id)
