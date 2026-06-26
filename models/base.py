from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Prediction:
    boxes: list[Box]
    labels: list[str]
    scores: list[float]
    masks: list[Any] | None = None
    latency_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.boxes)


class DetectionSegModel(ABC):
    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Prediction:
        ...
