from __future__ import annotations

import time
from typing import Any

from models.base import DetectionSegModel, Prediction


class Sam2Wrapper(DetectionSegModel):
    def __init__(self, weights: str = "sam2.1_t.pt", device: str | None = None) -> None:
        self.weights = weights
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM  # lazy import

            self._model = SAM(self.weights)
        return self._model

    def predict(
        self,
        image: Any,
        points: Any = None,
        bboxes: Any = None,
        **kwargs: Any,
    ) -> Prediction:
        if points is None and bboxes is None:
            raise ValueError("SAM 2 requires a prompt: pass `points` or `bboxes`.")

        model = self._load()
        start = time.perf_counter()
        results = model(image, points=points, bboxes=bboxes, device=self.device, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        masks = result.masks.data.tolist() if result.masks is not None else None

        return Prediction(
            boxes=[],
            labels=[],
            scores=[],
            masks=masks,
            latency_ms=latency_ms,
        )
