from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction

# The 80 COCO "things" class names, in category order.
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

# RF-DETR emits COCO 91-class *paper category ids* (1..90 with gaps), NOT a contiguous
# 0..79 index. Verified empirically: on bus.jpg it returns class_id [1, 6] for
# person + bus (91-scheme), which the old 0-indexed lookup mislabelled as bicycle + train.
_COCO91_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_NAME = dict(zip(_COCO91_IDS, COCO_CLASSES))


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
        labels = [COCO_ID_TO_NAME.get(int(i), str(int(i))) for i in det.class_id]

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            latency_ms=latency_ms,
        )
