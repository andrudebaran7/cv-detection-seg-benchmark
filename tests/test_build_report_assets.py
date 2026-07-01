import csv

from benchmark import build_report_assets as bra

_COLS = ["device", "model", "task", "image", "resolution",
         "experiment", "metric", "value", "n_iters", "measured_at"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        w.writerows(rows)


_DIST_COLS = ["device", "model", "task", "image_id", "latency_ms", "resolution", "measured_at"]


def _write_dist(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_DIST_COLS)
        w.writerows(rows)


def test_build_writes_feasibility_and_distribution_assets(tmp_path):
    cpu = tmp_path / "cpu.csv"
    mem = ["set", 640, "peak_rss", "rss_mb", 468.0, 1, "2026-07-01"]
    lat = ["set", 640, "warm_latency", "mean_ms", 200.0, 50, "2026-07-01"]
    _write(cpu, [["cpu", "yolo11n", "detection", *mem],
                 ["cpu", "yolo11n", "detection", *lat]])
    dist = tmp_path / "results_dist_cpu.csv"
    _write_dist(dist, [["cpu", "yolo11n", "detection", f"{i:012d}", 200.0 + i, 640, "2026-07-01"]
                       for i in range(10)])
    figd = tmp_path / "figures"
    texd = tmp_path / "generated"
    bra.build(cpu, None, figd, texd, dist_cpu=dist, dist_cuda=None)
    assert (texd / "feasibility_table.tex").exists()
    assert (texd / "distribution_table.tex").exists()
    assert (figd / "latency_boxplot.pdf").exists()
    assert "Yes" in (texd / "feasibility_table.tex").read_text()


def test_build_writes_figures_and_tables(tmp_path):
    cpu = tmp_path / "cpu.csv"
    cuda = tmp_path / "cuda.csv"
    base = ["set", 640, "warm_latency", "mean_ms", 1.0, 10, "2026-06-30"]
    _write(cpu, [["cpu", "yolo11n", "detection", *base],
                 ["cpu", "sam2-tiny", "segmentation", *base]])
    _write(cuda, [["cuda", "yolo11n", "detection", *base],
                  ["cuda", "sam2-tiny", "segmentation", *base]])
    figd = tmp_path / "figures"
    texd = tmp_path / "generated"
    bra.build(cpu, cuda, figd, texd)
    assert (figd / "cpu_vs_gpu.pdf").exists()
    assert (texd / "perf_det_table.tex").exists()
    assert (texd / "perf_seg_table.tex").exists()
    assert (texd / "memory_table.tex").exists()
    assert "\\begin{tabularx}" in (texd / "perf_det_table.tex").read_text()
