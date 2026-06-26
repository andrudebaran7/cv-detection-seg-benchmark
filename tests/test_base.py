import pytest
from models.base import Box, Prediction, DetectionSegModel


def test_prediction_len_counts_boxes():
    p = Prediction(
        boxes=[Box(0, 0, 1, 1), Box(2, 2, 3, 3)],
        labels=["cat", "dog"],
        scores=[0.9, 0.8],
        latency_ms=12.5,
    )
    assert len(p) == 2
    assert p.masks is None
    assert p.latency_ms == 12.5


def test_detectionsegmodel_is_abstract():
    with pytest.raises(TypeError):
        DetectionSegModel()
