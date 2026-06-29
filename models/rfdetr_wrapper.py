from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction

# 80 COCO class names, index-aligned with RF-DETR class_id.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class RfDetrWrapper(DetectionSegModel):
    def __init__(self, model_name: str = "nano", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import rfdetr  # lazy import

            kwargs = {"device": self.device} if self.device is not None else {}
            self._model = rfdetr.RFDETRNano(**kwargs)
        return self._model

    def predict(self, image: Any, threshold: float = 0.5, **kwargs: Any) -> Prediction:
        model = self._load()
        start = time.perf_counter()
        det = model.predict(image, threshold=threshold)
        latency_ms = (time.perf_counter() - start) * 1000.0

        boxes = [Box(float(x1), float(y1), float(x2), float(y2))
                 for x1, y1, x2, y2 in det.xyxy]
        scores = [float(c) for c in det.confidence]
        labels = [COCO_CLASSES[int(i)] if 0 <= int(i) < len(COCO_CLASSES) else str(int(i))
                  for i in det.class_id]

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            latency_ms=latency_ms,
        )
