from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from models.base import Prediction
from app.components.visualization import draw_prediction


@dataclass
class ComparisonResult:
    name: str
    prediction: Prediction
    image: Any  # PIL.Image.Image


def run_comparison(
    image: Any,
    specs: "list[tuple[str, str]]",
    loader: Callable[[str], Any],
    per_model_kwargs: "dict[str, dict] | None" = None,
) -> "list[ComparisonResult]":
    """Run several detectors on one image, one model at a time.

    `specs` is a list of ``(display_name, model_key)``. `loader(key)` returns the
    model for that key; it is expected to be the single-slot loader, so each call
    evicts the previously loaded model and only one is resident at a time.
    """
    per_model_kwargs = per_model_kwargs or {}
    results: list[ComparisonResult] = []
    for name, key in specs:
        model = loader(key)
        kwargs = per_model_kwargs.get(name, {})
        prediction = model.predict(image, **kwargs)
        rendered = draw_prediction(image, prediction)
        results.append(ComparisonResult(name=name, prediction=prediction, image=rendered))
    return results
