from unittest.mock import MagicMock

from models.base import Box, Prediction
from app.components.benchmark import (
    PUBLISHED_MAP,
    measure_latency,
    build_benchmark_rows,
)


def test_published_map_has_three_detectors():
    assert set(PUBLISHED_MAP) == {"YOLO11n", "RF-DETR-nano", "YOLO-World"}
    for entry in PUBLISHED_MAP.values():
        assert "map" in entry and "note" in entry


def test_measure_latency_returns_prediction_latency():
    model = MagicMock()
    model.predict.return_value = Prediction(
        boxes=[Box(0, 0, 1, 1)], labels=["x"], scores=[0.5], latency_ms=12.0
    )
    assert measure_latency(model, object(), conf=0.3) == 12.0
    model.predict.assert_called_once()


def test_build_benchmark_rows_joins_and_preserves_order():
    measured = {"RF-DETR-nano": 30.0, "YOLO11n": 10.0, "Unknown": 99.0}
    rows = build_benchmark_rows(measured)

    assert [r["model"] for r in rows] == ["RF-DETR-nano", "YOLO11n"]  # Unknown dropped
    assert rows[0]["latency_ms"] == 30.0
    assert rows[1]["latency_ms"] == 10.0
    assert rows[0]["map"] == PUBLISHED_MAP["RF-DETR-nano"]["map"]
