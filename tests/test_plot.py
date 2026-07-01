import csv

from benchmark import plot


def _write_rows(path):
    cols = ["device", "model", "task", "image", "resolution",
            "experiment", "metric", "value", "n_iters", "measured_at"]
    rows = [
        ["cpu", "yolo11n", "detection", "set", 640, "warm_latency", "mean_ms", 200.0, 50, "2026-06-29"],
        ["cpu", "yolo11n", "detection", "set", 320, "warm_latency", "mean_ms", 90.0, 50, "2026-06-29"],
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def test_load_rows_parses_numeric_fields(tmp_path):
    p = tmp_path / "r.csv"
    _write_rows(p)
    rows = plot.load_rows(p)
    assert rows[0]["value"] == 200.0
    assert rows[0]["resolution"] == 640


def test_plot_scaling_writes_file(tmp_path):
    p = tmp_path / "r.csv"
    _write_rows(p)
    out = tmp_path / "scaling.png"
    plot.plot_scaling(plot.load_rows(p), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_memory_scaling_writes_file(tmp_path):
    rows = [
        {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 320,
         "experiment": "peak_rss", "metric": "rss_mb", "value": 430.0},
        {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 640,
         "experiment": "peak_rss", "metric": "rss_mb", "value": 468.0},
    ]
    out = tmp_path / "mem.png"
    plot.plot_memory_scaling(rows, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_latency_boxplot_writes_file(tmp_path):
    rows = []
    for dev, base in (("cpu", 200.0), ("cuda", 9.0)):
        for i in range(10):
            rows.append({"device": dev, "model": "yolo11n", "task": "detection",
                         "latency_ms": base + i, "resolution": 640})
    out = tmp_path / "box.png"
    plot.plot_latency_boxplot(rows, out)
    assert out.exists() and out.stat().st_size > 0
