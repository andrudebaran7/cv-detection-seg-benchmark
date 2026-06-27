"""Single-slot model loader.

To stay within Streamlit Community Cloud's ~1 GB RAM, at most ONE model is held
in memory at a time. Requesting a different model releases the previous one
(drop the reference + gc) before loading the new one, so peak memory is the size
of the largest single model rather than the sum of all of them.
"""

import ctypes
import gc

import streamlit as st

from models.yolo_wrapper import YoloWrapper
from models.sam2_wrapper import Sam2Wrapper
from models.rfdetr_wrapper import RfDetrWrapper
from models.yoloworld_wrapper import YoloWorldWrapper
from models.mask2former_wrapper import Mask2FormerWrapper

# key -> zero-arg factory that builds a fresh wrapper.
_FACTORIES = {
    "yolo": lambda: YoloWrapper("yolo11n.pt"),
    "yolo_seg": lambda: YoloWrapper("yolo11n-seg.pt"),
    "sam2": lambda: Sam2Wrapper("sam2.1_t.pt"),
    "rfdetr": lambda: RfDetrWrapper("nano"),
    "yoloworld": lambda: YoloWorldWrapper("yolov8s-world.pt"),
    "mask2former": lambda: Mask2FormerWrapper(),
}


def _release_memory() -> None:
    """Return freed heap back to the OS, not just to the torch/glibc free pools.

    Dropping a model reference + ``gc.collect()`` alone does NOT lower RSS: CPU
    torch tensors and glibc retain freed memory in their own pools, so loading
    several models in a row keeps climbing and eventually OOMs. ``malloc_trim``
    forces glibc to give those pages back, which (measured) brings RSS back down
    to the framework floor after each eviction.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # non-glibc platform; gc.collect() above is the best we can do


def _load_into_slot(slot: dict, key: str, factories: dict):
    """Load `key`'s model into the single-entry `slot`, evicting any other."""
    if slot.get("key") != key:
        slot["model"] = None  # release the previous model before allocating the next
        slot["key"] = None
        _release_memory()
        slot["model"] = factories[key]()
        slot["key"] = key
    return slot["model"]


@st.cache_resource
def _slot() -> dict:
    """Persistent single-entry holder shared across reruns."""
    return {"key": None, "model": None}


def load_model(key: str):
    if key not in _FACTORIES:
        raise KeyError(f"unknown model key: {key}")
    return _load_into_slot(_slot(), key, _FACTORIES)


# Convenience accessors for single-model pages. Each routes through the single
# slot, so switching pages/models keeps only one model resident.
def get_yolo() -> YoloWrapper:
    return load_model("yolo")


def get_yolo_seg() -> YoloWrapper:
    return load_model("yolo_seg")


def get_sam2() -> Sam2Wrapper:
    return load_model("sam2")


def get_rfdetr() -> RfDetrWrapper:
    return load_model("rfdetr")


def get_yoloworld() -> YoloWorldWrapper:
    return load_model("yoloworld")


def get_mask2former() -> Mask2FormerWrapper:
    return load_model("mask2former")
