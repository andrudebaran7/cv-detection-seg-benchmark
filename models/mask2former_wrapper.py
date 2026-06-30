from __future__ import annotations

import time
from typing import Any

import numpy as np

from models.base import DetectionSegModel, Prediction


class Mask2FormerWrapper(DetectionSegModel):
    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-tiny-coco-panoptic",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import (  # lazy import
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )

            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
            if self.device is not None:
                model.to(self.device)
            self._model = (processor, model)
        return self._model

    def predict(self, image: Any, **kwargs: Any) -> Prediction:
        processor, model = self._load()
        pil = image if hasattr(image, "size") else None
        width, height = (pil.size if pil is not None else (image.shape[1], image.shape[0]))

        start = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt")
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = model(**inputs)
        result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        seg_obj = result["segmentation"]
        if hasattr(seg_obj, "cpu"):  # move a CUDA/torch tensor to host before numpy conversion
            seg_obj = seg_obj.cpu()
        seg = np.array(seg_obj)
        labels: list[str] = []
        scores: list[float] = []
        masks: list[Any] = []
        for info in result["segments_info"]:
            masks.append((seg == info["id"]))
            labels.append(model.config.id2label[info["label_id"]])
            scores.append(float(info["score"]))

        return Prediction(
            boxes=[],
            labels=labels,
            scores=scores,
            masks=masks,
            latency_ms=latency_ms,
        )
