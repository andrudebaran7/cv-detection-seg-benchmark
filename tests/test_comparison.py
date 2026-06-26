from unittest.mock import MagicMock

from PIL import Image

from models.base import Box, Prediction
from app.components.comparison import run_comparison, ComparisonResult


def _model_returning(pred):
    m = MagicMock()
    m.predict = MagicMock(return_value=pred)
    return m


def test_run_comparison_preserves_order_and_renders():
    img = Image.new("RGB", (40, 30), "white")
    pred_a = Prediction(boxes=[Box(1, 1, 5, 5)], labels=["a"], scores=[0.9], latency_ms=3.0)
    pred_b = Prediction(boxes=[], labels=[], scores=[], latency_ms=1.0)

    models = {"A": _model_returning(pred_a), "B": _model_returning(pred_b)}
    results = run_comparison(img, models)

    assert [r.name for r in results] == ["A", "B"]
    assert all(isinstance(r, ComparisonResult) for r in results)
    assert results[0].prediction is pred_a
    assert isinstance(results[0].image, Image.Image)
    assert results[0].image.size == (40, 30)


def test_run_comparison_passes_per_model_kwargs():
    img = Image.new("RGB", (10, 10), "white")
    pred = Prediction(boxes=[], labels=[], scores=[], latency_ms=0.0)
    model = _model_returning(pred)

    run_comparison(img, {"W": model}, per_model_kwargs={"W": {"classes": ["dog"]}})

    model.predict.assert_called_once()
    _, kwargs = model.predict.call_args
    assert kwargs.get("classes") == ["dog"]
