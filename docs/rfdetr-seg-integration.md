# Integrating RF-DETR-Seg with `DetectionSegModel`

> **Status:** Draft / design note. Generated as a research draft (Goose + `google/gemini-2.5-flash`, 2026-06-27) from the current `models/` interface. Review before implementing — it is a starting point, not validated code.

To integrate RF-DETR-Seg, create a new class `RFDETRSegWrapper` (or a similar name) that inherits from `DetectionSegModel`. This class encapsulates the RF-DETR-Seg model and adapts its output to the `Prediction` interface.

## 1. Class structure

```python
from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction

# Define COCO_CLASSES or similar for your specific model if needed
# COCO_CLASSES = [...]

class RFDETRSegWrapper(DetectionSegModel):
    def __init__(self, model_name: str = "your_model_variant") -> None:
        self.model_name = model_name
        self._model = None  # To be loaded lazily

    def _load(self):
        if self._model is None:
            # Import your RF-DETR-Seg specific library here
            # e.g., from rfdetrseg import RFDETRSegModel
            # self._model = RFDETRSegModel(self.model_name)
            pass  # Placeholder
        return self._model

    def predict(self, image: Any, threshold: float = 0.5, **kwargs: Any) -> Prediction:
        model = self._load()
        start = time.perf_counter()
        # Your RF-DETR-Seg specific prediction call
        # e.g., detr_output = model.predict(image, threshold=threshold)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # ... (populate boxes, labels, scores, and masks) ...

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=masks,
            latency_ms=latency_ms,
        )
```

## 2. Methods to implement

- **`__init__(self, model_name: str = "your_model_variant")`** — initialize the model. Use lazy loading (as in `_load`) to defer creation until the first prediction, which saves memory when the model is not always used.
- **`_load(self)`** — helper for lazy loading the RF-DETR-Seg model.
- **`predict(self, image, threshold=0.5, **kwargs) -> Prediction`** — the core inference method. It should:
    - Accept an `image` (format defined by the concrete RF-DETR-Seg implementation) and a `threshold` for confidence.
    - Call the RF-DETR-Seg model's prediction method.
    - Measure inference `latency_ms`.
    - Transform the raw output into a `Prediction` instance.

## 3. Populating the `masks` field

RF-DETR-Seg models typically output segmentation masks directly. Extract them and ensure they match the `list[Any]` shape of `Prediction.masks`. The `YoloWrapper` example assigns `result.masks.data.tolist()`; adapt this to RF-DETR-Seg's mask output:

- If the model outputs binary masks (e.g. `(H, W)` arrays per instance), include them directly in the list.
- If it predicts mask parameters (e.g. coefficients for polynomial curves or implicit functions), render them into binary masks before adding them to the `Prediction`.

## Consistency notes (vs. existing wrappers)

- Follow `RfDetrWrapper` for the detection path: build `boxes` with `Box(float(x1), float(y1), float(x2), float(y2))`, map `class_id` to label names, cast `confidence` to `float`.
- The only addition over `RfDetrWrapper` is populating `masks` instead of leaving it `None`.
