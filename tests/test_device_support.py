# tests/test_device_support.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np


def _fake_ultralytics_yolo():
    mod = types.ModuleType("ultralytics")
    result = MagicMock()
    result.names = {0: "person"}
    result.boxes.xyxy.tolist.return_value = [[1.0, 2.0, 3.0, 4.0]]
    result.boxes.conf.tolist.return_value = [0.9]
    result.boxes.cls.tolist.return_value = [0]
    result.masks = None
    instance = MagicMock(return_value=[result])
    mod.YOLO = MagicMock(return_value=instance)
    sys.modules["ultralytics"] = mod
    return instance


def test_yolo_threads_device_into_call():
    instance = _fake_ultralytics_yolo()
    from models.yolo_wrapper import YoloWrapper

    YoloWrapper(device="cuda").predict(np.zeros((8, 8, 3), dtype=np.uint8))
    # the model was called with device="cuda"
    _, kwargs = instance.call_args
    assert kwargs.get("device") == "cuda"


def test_yolo_default_device_is_none_call_omits_or_none():
    instance = _fake_ultralytics_yolo()
    from models.yolo_wrapper import YoloWrapper

    YoloWrapper().predict(np.zeros((8, 8, 3), dtype=np.uint8))
    _, kwargs = instance.call_args
    assert kwargs.get("device") is None


def test_mask2former_moves_model_to_device():
    mod = types.ModuleType("transformers")
    processor = MagicMock()
    processor.return_value = MagicMock()  # inputs object
    model = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    mod.AutoImageProcessor.from_pretrained = MagicMock(return_value=processor)
    mod.Mask2FormerForUniversalSegmentation = MagicMock()
    mod.Mask2FormerForUniversalSegmentation.from_pretrained = MagicMock(return_value=model)
    sys.modules["transformers"] = mod

    from models.mask2former_wrapper import Mask2FormerWrapper

    w = Mask2FormerWrapper(device="cuda")
    w._load()
    model.to.assert_called_with("cuda")
