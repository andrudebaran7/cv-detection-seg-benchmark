from app.components.model_runner import _load_into_slot, _FACTORIES, load_model


def test_load_into_slot_holds_one_model_at_a_time():
    slot = {"key": None, "model": None}
    factories = {"a": lambda: "MODEL_A", "b": lambda: "MODEL_B"}

    m1 = _load_into_slot(slot, "a", factories)
    assert m1 == "MODEL_A"
    assert slot["key"] == "a"

    m2 = _load_into_slot(slot, "b", factories)
    assert m2 == "MODEL_B"
    assert slot["key"] == "b"
    # Only the most recent model is held in the slot.
    assert slot["model"] == "MODEL_B"


def test_load_into_slot_does_not_rebuild_same_key():
    builds = []
    factories = {"a": lambda: builds.append(1) or "A"}
    slot = {"key": "a", "model": "EXISTING"}

    result = _load_into_slot(slot, "a", factories)

    assert result == "EXISTING"
    assert builds == []  # cached model reused, factory not called


def test_factory_registry_has_all_model_keys():
    assert set(_FACTORIES) == {
        "yolo", "yolo_seg", "sam2", "rfdetr", "yoloworld", "mask2former"
    }


def test_load_model_rejects_unknown_key():
    import pytest
    with pytest.raises(KeyError):
        load_model("does-not-exist")
