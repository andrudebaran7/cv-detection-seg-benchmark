from benchmark import manifest


def test_manifest_has_required_keys_and_device():
    m = manifest.build_manifest("cpu")
    for key in ["device", "cpu", "ram_gb", "gpu", "torch_version",
                "cuda_available", "python", "os", "measured_at"]:
        assert key in m
    assert m["device"] == "cpu"
    assert m["ram_gb"] > 0.0
