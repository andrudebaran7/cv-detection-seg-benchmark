import csv

from benchmark import combine

_COLS = ["device", "model", "task", "image", "resolution",
         "experiment", "metric", "value", "n_iters", "measured_at"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        w.writerows(rows)


def test_load_combined_merges_and_coerces(tmp_path):
    _write(tmp_path / "cpu.csv", [
        ["cpu", "yolo11n", "detection", "set", 640, "warm_latency", "mean_ms", 260.0, 10, "2026-06-30"]])
    _write(tmp_path / "cuda.csv", [
        ["cuda", "yolo11n", "detection", "set", 640, "warm_latency", "mean_ms", 9.0, 50, "2026-06-30"]])
    rows = combine.load_combined(tmp_path / "cpu.csv", tmp_path / "cuda.csv")
    assert len(rows) == 2
    assert {r["device"] for r in rows} == {"cpu", "cuda"}
    assert rows[0]["resolution"] == 640 and isinstance(rows[0]["value"], float)


def test_value_for_and_models_for_task(tmp_path):
    _write(tmp_path / "cpu.csv", [
        ["cpu", "yolo11n", "detection", "set", 640, "warm_latency", "mean_ms", 260.0, 10, "2026-06-30"],
        ["cpu", "sam2-tiny", "segmentation", "set", 640, "warm_latency", "mean_ms", 5815.0, 10, "2026-06-30"]])
    rows = combine.load_combined(tmp_path / "cpu.csv")
    assert combine.value_for(rows, device="cpu", model="yolo11n",
                             experiment="warm_latency", metric="mean_ms", resolution=640) == 260.0
    assert combine.value_for(rows, device="cuda", model="yolo11n",
                             experiment="warm_latency", metric="mean_ms", resolution=640) is None
    assert combine.models_for_task(rows, "detection") == ["yolo11n"]
