import csv

from PIL import Image

from benchmark import run
from benchmark.models_registry import ModelSpec


class _FakeModel:
    def __init__(self, device=None):
        self.device = device
    def predict(self, image, **kwargs):
        class P:  # minimal Prediction-like
            latency_ms = 1.0
        return P()


def _fake_spec():
    return ModelSpec("fake", "detection", lambda device: _FakeModel(device), lambda img: {})


def test_run_model_emits_rows_for_each_experiment():
    img = Image.new("RGB", (640, 480))
    rows = run.run_model(_fake_spec(), img, device="cpu", resolution=640, iters=3, warmup=1)
    experiments = {r["experiment"] for r in rows}
    assert {"warm_latency", "cold_start", "throughput", "peak_rss"} <= experiments
    for r in rows:
        assert r["device"] == "cpu"
        assert r["model"] == "fake"
        assert r["resolution"] == 640
        assert isinstance(r["value"], float)


def test_write_csv_has_fixed_header(tmp_path):
    rows = [{"device": "cpu", "model": "fake", "task": "detection", "image": "x",
             "resolution": 640, "experiment": "warm_latency", "metric": "mean_ms",
             "value": 1.0, "n_iters": 3, "measured_at": "2026-06-29"}]
    p = tmp_path / "r.csv"
    run.write_csv(rows, p)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["device", "model", "task", "image", "resolution",
                      "experiment", "metric", "value", "n_iters", "measured_at"]


def test_main_writes_csv_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {"fake": _fake_spec()})
    monkeypatch.setattr(run, "load_images", lambda: [Image.new("RGB", (64, 64))])
    monkeypatch.setattr(run, "RESOLUTIONS", [320])
    run.main(["--device", "cpu", "--models", "all", "--iters", "2",
              "--warmup", "1", "--out", str(tmp_path)])
    assert (tmp_path / "results_cpu.csv").exists()
    assert (tmp_path / "manifest_cpu.json").exists()


def _boom_spec():
    def _factory(device):
        raise RuntimeError("model load failed (simulated transient)")
    return ModelSpec("boom", "detection", _factory, lambda img: {})


class _FlakyModel:
    """Succeeds at the first resolution, then raises — to exercise mid-sweep failure."""
    seen_sizes: set = set()

    def __init__(self, device=None):
        self.device = device

    def predict(self, image, **kwargs):
        if image.size[0] >= 640:
            raise RuntimeError("simulated mid-sweep failure at higher resolution")
        class P:
            latency_ms = 1.0
        return P()


def _flaky_spec():
    return ModelSpec("flaky", "detection", lambda device: _FlakyModel(device), lambda img: {})


def test_main_isolates_per_model_failure(tmp_path, monkeypatch):
    # One failing model must not abort the campaign: the good model's rows are
    # still written, and the run does not raise.
    monkeypatch.setattr(run, "REGISTRY", {"boom": _boom_spec(), "fake": _fake_spec()})
    monkeypatch.setattr(run, "load_images", lambda: [Image.new("RGB", (64, 64))])
    monkeypatch.setattr(run, "RESOLUTIONS", [320])
    run.main(["--device", "cpu", "--models", "all", "--iters", "2",
              "--warmup", "1", "--out", str(tmp_path)])
    out_csv = tmp_path / "results_cpu.csv"
    assert out_csv.exists()
    with open(out_csv) as f:
        models = {row["model"] for row in csv.DictReader(f)}
    assert models == {"fake"}  # boom skipped, fake survived


def test_main_rolls_back_partial_rows_on_midsweep_failure(tmp_path, monkeypatch):
    # A model that runs at 320 but fails at 640 must leave NO rows (skipped == no data).
    monkeypatch.setattr(run, "REGISTRY", {"flaky": _flaky_spec(), "fake": _fake_spec()})
    monkeypatch.setattr(run, "load_images", lambda: [Image.new("RGB", (64, 64))])
    monkeypatch.setattr(run, "RESOLUTIONS", [320, 640])
    run.main(["--device", "cpu", "--models", "all", "--iters", "2",
              "--warmup", "1", "--out", str(tmp_path)])
    with open(tmp_path / "results_cpu.csv") as f:
        models = {row["model"] for row in csv.DictReader(f)}
    assert models == {"fake"}  # flaky's partial 320 rows rolled back
