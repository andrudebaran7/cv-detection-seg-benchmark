from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction


class YoloWrapper(DetectionSegModel):
    def __init__(self, weights: str = "yolo11n.pt") -> None:
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy import

            self._model = YOLO(self.weights)
        return self._model

    def predict(self, image: Any, conf: float = 0.25, **kwargs: Any) -> Prediction:
        model = self._load()
        start = time.perf_counter()
        results = model(image, conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()
        names = result.names

        boxes = [Box(*coords) for coords in xyxy]
        labels = [names[int(c)] for c in cls_ids]
        scores = [float(s) for s in confs]
        masks = result.masks.data.tolist() if result.masks is not None else None

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=masks,
            latency_ms=latency_ms,
        )
