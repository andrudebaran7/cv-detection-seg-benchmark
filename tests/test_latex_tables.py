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


_DIST_ROWS = [
    {"device": "cpu", "model": "yolo11n", "task": "detection",
     "latency_ms": 200.0, "resolution": 640},
    {"device": "cpu", "model": "yolo11n", "task": "detection",
     "latency_ms": 220.0, "resolution": 640},
    {"device": "cuda", "model": "yolo11n", "task": "detection",
     "latency_ms": 9.0, "resolution": 640},
    {"device": "cuda", "model": "yolo11n", "task": "detection",
     "latency_ms": 11.0, "resolution": 640},
]

_MEM_ROWS = [
    {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 640,
     "experiment": "peak_rss", "metric": "rss_mb", "value": 468.0},
    {"device": "cpu", "model": "sam2-tiny", "task": "segmentation", "resolution": 640,
     "experiment": "peak_rss", "metric": "rss_mb", "value": 1600.0},
]


def test_distribution_table_has_percentiles_and_devices():
    tex = lt.distribution_table(_DIST_ROWS, "tab:latency-dist")
    assert "\\label{tab:latency-dist}" in tex
    assert "yolo11n" in tex
    assert "cpu" in tex and "cuda" in tex
    assert "P50" in tex and "P90" in tex and "P99" in tex
    assert "\\begin{tabularx}" in tex and "\\toprule" in tex


def test_feasibility_table_marks_fit_and_oom():
    tex = lt.feasibility_table(_MEM_ROWS, "tab:feasibility")
    assert "\\label{tab:feasibility}" in tex
    assert "468" in tex and "Yes" in tex        # yolo11n fits
    assert "1600" in tex and "No" in tex        # sam2-tiny OOM


def test_feasibility_table_renders_dash_for_missing_rss():
    # A model with a row but no peak_rss measurement -> RSS unknown -> "--".
    rows = [{"device": "cpu", "model": "ghost", "task": "detection", "resolution": 640,
             "experiment": "warm_latency", "metric": "mean_ms", "value": 100.0}]
    tex = lt.feasibility_table(rows, "tab:feasibility")
    assert "ghost & -- &" in tex
