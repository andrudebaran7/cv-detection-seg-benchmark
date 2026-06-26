import sys
import types
from unittest.mock import MagicMock

import numpy as np

from models.base import Prediction


def _install_fake_ultralytics(fake_result):
    """Inject a fake `ultralytics` module exposing YOLO -> callable returning [fake_result]."""
    mod = types.ModuleType("ultralytics")
    model_instance = MagicMock(return_value=[fake_result])
    mod.YOLO = MagicMock(return_value=model_instance)
    sys.modules["ultralytics"] = mod
    return mod


def _make_fake_result():
    r = MagicMock()
    r.names = {0: "person", 16: "dog"}
    r.boxes.xyxy.tolist.return_value = [[10.0, 20.0, 30.0, 40.0]]
    r.boxes.conf.tolist.return_value = [0.91]
    r.boxes.cls.tolist.return_value = [0]
    r.masks = None
    return r


def test_yolo_predict_maps_boxes_labels_scores(monkeypatch):
    _install_fake_ultralytics(_make_fake_result())
    from models.yolo_wrapper import YoloWrapper

    pred = YoloWrapper().predict(np.zeros((64, 64, 3), dtype=np.uint8))

    assert isinstance(pred, Prediction)
    assert len(pred) == 1
    assert pred.labels == ["person"]
    assert pred.scores == [0.91]
    assert pred.boxes[0].x1 == 10.0 and pred.boxes[0].y2 == 40.0
    assert pred.latency_ms >= 0.0


def test_yolo_lazy_import_not_at_module_top():
    # Importing the wrapper module must not require ultralytics.
    sys.modules.pop("ultralytics", None)
    import importlib
    import models.yolo_wrapper as yw
    importlib.reload(yw)  # should not raise
