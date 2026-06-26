import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from models.base import Prediction


def _install_fake_transformers():
    mod = types.ModuleType("transformers")

    processor = MagicMock()
    # processor(images=..., return_tensors=...) -> dict-like inputs
    processor.return_value = {"pixel_values": MagicMock()}
    # post_process returns one dict: a 2x2 segmentation map with ids 1 and 2
    seg = np.array([[1, 1], [2, 2]])
    processor.post_process_panoptic_segmentation.return_value = [
        {
            "segmentation": seg,
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.9},
                {"id": 2, "label_id": 5, "score": 0.8},
            ],
        }
    ]

    model = MagicMock()
    model.config.id2label = {0: "person", 5: "bus"}
    model.return_value = MagicMock()  # outputs

    mod.AutoImageProcessor = MagicMock()
    mod.AutoImageProcessor.from_pretrained = MagicMock(return_value=processor)
    mod.Mask2FormerForUniversalSegmentation = MagicMock()
    mod.Mask2FormerForUniversalSegmentation.from_pretrained = MagicMock(return_value=model)
    sys.modules["transformers"] = mod
    return processor, model


def test_mask2former_maps_panoptic_to_prediction():
    _install_fake_transformers()
    from models.mask2former_wrapper import Mask2FormerWrapper

    img = Image.new("RGB", (2, 2), "white")
    pred = Mask2FormerWrapper().predict(img)

    assert isinstance(pred, Prediction)
    assert pred.boxes == []
    assert pred.labels == ["person", "bus"]
    assert pred.scores == [0.9, 0.8]
    assert pred.masks is not None and len(pred.masks) == 2
    # first mask selects the top row (id == 1)
    assert np.array(pred.masks[0]).tolist() == [[True, True], [False, False]]
    assert pred.latency_ms >= 0.0


def test_mask2former_lazy_import():
    sys.modules.pop("transformers", None)
    import importlib
    import models.mask2former_wrapper as mw
    importlib.reload(mw)  # importing must not require transformers
