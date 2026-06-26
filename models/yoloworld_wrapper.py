from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction


class YoloWorldWrapper(DetectionSegModel):
    def __init__(self, weights: str = "yolov8s-world.pt") -> None:
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLOWorld  # lazy import

            self._model = YOLOWorld(self.weights)
        return self._model

    def predict(
        self,
        image: Any,
        classes: list[str] | None = None,
        conf: float = 0.25,
        **kwargs: Any,
    ) -> Prediction:
        if not classes:
            raise ValueError("YOLO-World requires at least one text class.")

        model = self._load()
        model.set_classes(classes)

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

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            latency_ms=latency_ms,
        )
