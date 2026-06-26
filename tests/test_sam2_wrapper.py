import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from models.base import Prediction


def _install_fake_ultralytics_sam(fake_result):
    mod = types.ModuleType("ultralytics")
    model_instance = MagicMock(return_value=[fake_result])
    mod.SAM = MagicMock(return_value=model_instance)
    sys.modules["ultralytics"] = mod


def _make_fake_sam_result():
    r = MagicMock()
    r.masks.data.tolist.return_value = [[[0, 1], [1, 0]]]
    return r


def test_sam2_predict_returns_masks(monkeypatch):
    _install_fake_ultralytics_sam(_make_fake_sam_result())
    from models.sam2_wrapper import Sam2Wrapper

    pred = Sam2Wrapper().predict(
        np.zeros((64, 64, 3), dtype=np.uint8), points=[[32, 32]]
    )
    assert isinstance(pred, Prediction)
    assert pred.masks == [[[0, 1], [1, 0]]]
    assert pred.boxes == []
    assert pred.latency_ms >= 0.0


def test_sam2_requires_a_prompt():
    _install_fake_ultralytics_sam(_make_fake_sam_result())
    from models.sam2_wrapper import Sam2Wrapper

    with pytest.raises(ValueError):
        Sam2Wrapper().predict(np.zeros((64, 64, 3), dtype=np.uint8))
