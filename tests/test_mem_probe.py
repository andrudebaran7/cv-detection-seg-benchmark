from PIL import Image

from benchmark import mem_probe
from benchmark.models_registry import ModelSpec


class _FakeModel:
    def __init__(self, device=None):
        self.device = device

    def predict(self, image, **kwargs):
        class P:
            latency_ms = 1.0
        return P()


def _fake_spec():
    return ModelSpec("fake", "detection", lambda d: _FakeModel(d), lambda img: {})


def test_measure_peak_memory_returns_rss_and_gpu(monkeypatch):
    # Runs in-process here (the isolation is provided by running this as a fresh
    # subprocess in the real campaign); checks the returned shape and that RSS is a peak > 0.
    monkeypatch.setattr(mem_probe, "REGISTRY", {"fake": _fake_spec()})
    monkeypatch.setattr(mem_probe, "load_images", lambda: [Image.new("RGB", (64, 64))])
    out = mem_probe.measure_peak_memory("fake", "cpu", 320, iters=2, warmup=1)
    assert set(out) == {"rss_mb", "gpu_mb"}
    assert out["rss_mb"] > 0.0       # ru_maxrss is a real peak, never zero
    assert out["gpu_mb"] == 0.0      # no CUDA in this test
