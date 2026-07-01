import sys
import types
from unittest.mock import MagicMock

import numpy as np

from models.base import Prediction


def _fake_detections():
    det = MagicMock()
    det.xyxy = np.array([[5.0, 6.0, 7.0, 8.0]])
    det.confidence = np.array([0.77])
    det.class_id = np.array([1])  # COCO 91-scheme paper id 1 -> "person"
    return det


def _install_fake_rfdetr(det):
    mod = types.ModuleType("rfdetr")
    instance = MagicMock()
    instance.predict = MagicMock(return_value=det)
    mod.RFDETRNano = MagicMock(return_value=instance)
    sys.modules["rfdetr"] = mod
    return mod


def test_rfdetr_predict_maps_to_prediction():
    _install_fake_rfdetr(_fake_detections())
    from models.rfdetr_wrapper import RfDetrWrapper

    pred = RfDetrWrapper().predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert isinstance(pred, Prediction)
    assert len(pred) == 1
    assert pred.boxes[0].x1 == 5.0 and pred.boxes[0].y2 == 8.0
    assert pred.scores == [0.77]
    assert pred.labels == ["person"]
    assert pred.masks is None
    assert pred.latency_ms >= 0.0


def test_rfdetr_lazy_import():
    sys.modules.pop("rfdetr", None)
    import importlib
    import models.rfdetr_wrapper as rw
    importlib.reload(rw)  # importing must not require rfdetr


def test_rfdetr_maps_coco91_ids_to_names():
    # Regression guard for the COCO 91-scheme mapping: class_id 1,6 -> person, bus
    # (the old 0-indexed lookup mislabelled these as bicycle, train).
    det = MagicMock()
    det.xyxy = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    det.confidence = np.array([0.9, 0.8])
    det.class_id = np.array([1, 6])
    _install_fake_rfdetr(det)
    from models.rfdetr_wrapper import RfDetrWrapper

    pred = RfDetrWrapper().predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert pred.labels == ["person", "bus"]
