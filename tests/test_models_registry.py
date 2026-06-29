from PIL import Image

from benchmark import models_registry as reg


def test_registry_has_six_models_with_tasks():
    assert set(reg.REGISTRY) == {
        "yolo11n", "yolo11n-seg", "sam2-tiny", "rfdetr-nano", "yolo-world", "mask2former",
    }
    tasks = {s.task for s in reg.REGISTRY.values()}
    assert tasks == {"detection", "segmentation"}


def test_sam2_predict_kwargs_supply_center_point():
    spec = reg.REGISTRY["sam2-tiny"]
    kwargs = spec.predict_kwargs(Image.new("RGB", (640, 480)))
    assert "points" in kwargs and kwargs["points"] == [[320, 240]]


def test_yoloworld_predict_kwargs_supply_classes():
    spec = reg.REGISTRY["yolo-world"]
    kwargs = spec.predict_kwargs(Image.new("RGB", (640, 480)))
    assert kwargs.get("classes")  # non-empty list of text classes


def test_factory_passes_device_through():
    import sys, types
    from unittest.mock import MagicMock
    mod = types.ModuleType("ultralytics")
    mod.YOLO = MagicMock(return_value=MagicMock())
    sys.modules["ultralytics"] = mod
    model = reg.REGISTRY["yolo11n"].factory("cuda")
    assert model.device == "cuda"
