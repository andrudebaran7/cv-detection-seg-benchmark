import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from models.base import Prediction


def _make_fake_result():
    r = MagicMock()
    r.names = {0: "cat", 1: "umbrella"}
    r.boxes.xyxy.tolist.return_value = [[1.0, 2.0, 3.0, 4.0]]
    r.boxes.conf.tolist.return_value = [0.66]
    r.boxes.cls.tolist.return_value = [1]
    r.masks = None
    return r


def _install_fake_yoloworld(fake_result):
    mod = types.ModuleType("ultralytics")
    instance = MagicMock(return_value=[fake_result])
    instance.set_classes = MagicMock()
    mod.YOLOWorld = MagicMock(return_value=instance)
    sys.modules["ultralytics"] = mod
    return instance


def test_yoloworld_sets_classes_and_maps_prediction():
    instance = _install_fake_yoloworld(_make_fake_result())
    from models.yoloworld_wrapper import YoloWorldWrapper

    pred = YoloWorldWrapper().predict(
        np.zeros((32, 32, 3), dtype=np.uint8), classes=["cat", "umbrella"]
    )

    instance.set_classes.assert_called_once_with(["cat", "umbrella"])
    assert isinstance(pred, Prediction)
    assert pred.labels == ["umbrella"]
    assert pred.scores == [0.66]
    assert pred.boxes[0].x1 == 1.0
    assert pred.latency_ms >= 0.0


def test_yoloworld_requires_classes():
    _install_fake_yoloworld(_make_fake_result())
    from models.yoloworld_wrapper import YoloWorldWrapper

    with pytest.raises(ValueError):
        YoloWorldWrapper().predict(np.zeros((32, 32, 3), dtype=np.uint8), classes=[])
