import csv

from PIL import Image

from benchmark import run_dist
from benchmark.models_registry import ModelSpec


class _FakeModel:
    def __init__(self, device=None):
        self.device = device

    def predict(self, image, **kwargs):
        class P:
            latency_ms = 1.0
        return P()


def _fake_spec():
    return ModelSpec("fake", "detection", lambda device: _FakeModel(device), lambda img: {})


def test_run_dist_model_one_row_per_image():
    images = [Image.new("RGB", (100, 80)) for _ in range(4)]
    ids = ["a", "b", "c", "d"]
    rows = run_dist.run_dist_model(_fake_spec(), images, ids, device="cpu", warmup=1)
    assert len(rows) == 4
    assert [r["image_id"] for r in rows] == ids
    for r in rows:
        assert r["device"] == "cpu" and r["model"] == "fake" and r["task"] == "detection"
        assert r["resolution"] == 640
        assert isinstance(r["latency_ms"], float)


def test_write_dist_csv_has_fixed_header(tmp_path):
    rows = [{"device": "cpu", "model": "fake", "task": "detection", "image_id": "a",
             "latency_ms": 1.0, "resolution": 640, "measured_at": "2026-07-01"}]
    p = tmp_path / "results_dist_cpu.csv"
    run_dist.write_dist_csv(rows, p)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["device", "model", "task", "image_id",
                      "latency_ms", "resolution", "measured_at"]


def test_main_writes_csv_and_manifest(tmp_path, monkeypatch):
    imgs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    paths = []
    for i, im in enumerate(imgs):
        pp = tmp_path / f"{i:012d}.jpg"
        im.save(pp)
        paths.append(pp)
    monkeypatch.setattr(run_dist.fetch_images, "fetch_dist_images", lambda limit=100: paths)
    monkeypatch.setattr(run_dist, "REGISTRY", {"fake": _fake_spec()})
    out = tmp_path / "phase2"
    run_dist.main(["--device", "cpu", "--out", str(out), "--limit", "3", "--warmup", "1"])
    assert (out / "results_dist_cpu.csv").exists()
    assert (out / "manifest_dist_cpu.json").exists()
    with open(out / "results_dist_cpu.csv") as f:
        data = list(csv.DictReader(f))
    assert len(data) == 3 and all(d["model"] == "fake" for d in data)
