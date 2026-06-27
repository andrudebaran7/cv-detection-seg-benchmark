from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Box:
    """Bounding box coordinates.

    Attributes:
        x1: Left (or minimum) x coordinate.
        y1: Top (or minimum) y coordinate.
        x2: Right (or maximum) x coordinate.
        y2: Bottom (or maximum) y coordinate.
    """

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Prediction:
    """Model prediction container.

    Attributes:
        boxes: List of detected bounding boxes.
        labels: Corresponding class labels for each box.
        scores: Confidence scores for each detection.
        masks: Optional segmentation masks; ``None`` if not applicable.
        latency_ms: Inference latency in milliseconds.
    """

    boxes: list[Box]
    labels: list[str]
    scores: list[float]
    masks: list[Any] | None = None
    latency_ms: float = 0.0

    def __len__(self) -> int:
        """Return the number of detected boxes.

        This allows ``len(prediction)`` to be used as a shortcut for
        ``len(prediction.boxes)``.
        """
        return len(self.boxes)


class DetectionSegModel(ABC):
    """Abstract base class for detection/segmentation models."""

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Prediction:
        """Run inference on an image.

        Args:
            image: Input image in any format accepted by the concrete model.
            **kwargs: Additional model‑specific parameters.

        Returns:
            A :class:`Prediction` instance containing detection/segmentation
            results and optional latency information.
        """
        ...
