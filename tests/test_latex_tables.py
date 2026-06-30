from benchmark import latex_tables as lt

_ROWS = [
    {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 640,
     "experiment": "warm_latency", "metric": "mean_ms", "value": 260.0},
    {"device": "cuda", "model": "yolo11n", "task": "detection", "resolution": 640,
     "experiment": "warm_latency", "metric": "mean_ms", "value": 9.0},
    {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 640,
     "experiment": "throughput", "metric": "imgs_per_sec", "value": 3.8},
]


def test_latency_table_has_models_and_devices_and_label():
    tex = lt.latency_table(_ROWS, "detection", "tab:perf-det")
    assert "\\label{tab:perf-det}" in tex
    assert "yolo11n" in tex
    assert "260" in tex and "9" in tex          # cpu and gpu latency rendered
    assert "\\begin{tabularx}" in tex and "\\toprule" in tex


def test_missing_value_renders_dash():
    rows = [r for r in _ROWS if r["device"] == "cpu"]  # no GPU row
    tex = lt.latency_table(rows, "detection", "tab:perf-det")
    assert "--" in tex  # GPU column shows a dash
