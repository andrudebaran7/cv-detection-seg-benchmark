from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from models.base import DetectionSegModel

_COCO_CLASSES = ["person", "car", "dog", "chair", "bottle"]  # text prompts for YOLO-World


@dataclass
class ModelSpec:
    key: str
    task: str
    factory: Callable[[str | None], DetectionSegModel]
    predict_kwargs: Callable[[Image.Image], dict]


def _center_point(img: Image.Image) -> dict:
    w, h = img.size
    return {"points": [[w // 2, h // 2]]}


def _yolo11n(device):
    from models.yolo_wrapper import YoloWrapper
    return YoloWrapper(device=device)


def _yolo11n_seg(device):
    from models.yolo_wrapper import YoloWrapper
    return YoloWrapper(weights="yolo11n-seg.pt", device=device)


def _sam2(device):
    from models.sam2_wrapper import Sam2Wrapper
    return Sam2Wrapper(device=device)


def _rfdetr(device):
    from models.rfdetr_wrapper import RfDetrWrapper
    return RfDetrWrapper(device=device)


def _yoloworld(device):
    from models.yoloworld_wrapper import YoloWorldWrapper
    return YoloWorldWrapper(device=device)


def _mask2former(device):
    from models.mask2former_wrapper import Mask2FormerWrapper
    return Mask2FormerWrapper(device=device)


REGISTRY: dict[str, ModelSpec] = {
    "yolo11n": ModelSpec("yolo11n", "detection", _yolo11n, lambda img: {}),
    "yolo11n-seg": ModelSpec("yolo11n-seg", "segmentation", _yolo11n_seg, lambda img: {}),
    "sam2-tiny": ModelSpec("sam2-tiny", "segmentation", _sam2, _center_point),
    "rfdetr-nano": ModelSpec("rfdetr-nano", "detection", _rfdetr, lambda img: {}),
    "yolo-world": ModelSpec("yolo-world", "detection", _yoloworld,
                            lambda img: {"classes": _COCO_CLASSES}),
    "mask2former": ModelSpec("mask2former", "segmentation", _mask2former, lambda img: {}),
}
