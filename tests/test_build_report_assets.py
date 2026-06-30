import csv

from benchmark import build_report_assets as bra

_COLS = ["device", "model", "task", "image", "resolution",
         "experiment", "metric", "value", "n_iters", "measured_at"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        w.writerows(rows)


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
