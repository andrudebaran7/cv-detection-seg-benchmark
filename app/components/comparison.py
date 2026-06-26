from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.base import DetectionSegModel, Prediction
from app.components.visualization import draw_prediction


@dataclass
class ComparisonResult:
    name: str
    prediction: Prediction
    image: Any  # PIL.Image.Image


def run_comparison(
    image: Any,
    models: "dict[str, DetectionSegModel]",
    per_model_kwargs: "dict[str, dict] | None" = None,
) -> "list[ComparisonResult]":
    per_model_kwargs = per_model_kwargs or {}
    results: list[ComparisonResult] = []
    for name, model in models.items():
        kwargs = per_model_kwargs.get(name, {})
        prediction = model.predict(image, **kwargs)
        rendered = draw_prediction(image, prediction)
        results.append(ComparisonResult(name=name, prediction=prediction, image=rendered))
    return results
