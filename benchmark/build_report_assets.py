from __future__ import annotations

import argparse
import pathlib

from benchmark import latex_tables as lt
from benchmark import plot
from benchmark.combine import load_combined


def build(cpu_csv, cuda_csv, fig_dir, tex_dir):
    fig_dir = pathlib.Path(fig_dir)
    tex_dir = pathlib.Path(tex_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for p in (cpu_csv, cuda_csv) if p and pathlib.Path(p).exists()]
    rows = load_combined(*paths)

    plot.plot_cpu_vs_gpu(rows, fig_dir / "cpu_vs_gpu.pdf")
    plot.plot_scaling(rows, fig_dir / "latency_scaling.pdf")
    plot.plot_memory_scaling(rows, fig_dir / "memory_scaling.pdf")

    (tex_dir / "perf_det_table.tex").write_text(lt.latency_table(rows, "detection", "tab:perf-det"))
    (tex_dir / "perf_seg_table.tex").write_text(lt.latency_table(rows, "segmentation", "tab:perf-seg"))
    (tex_dir / "coldwarm_det_table.tex").write_text(lt.coldwarm_table(rows, "detection", "tab:coldwarm-det"))
    (tex_dir / "coldwarm_seg_table.tex").write_text(lt.coldwarm_table(rows, "segmentation", "tab:coldwarm-seg"))
    (tex_dir / "memory_table.tex").write_text(lt.memory_table(rows, "tab:perf-mem"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build report figures + table fragments from CSVs")
    ap.add_argument("--cpu", default="data/phase2/results_cpu.csv")
    ap.add_argument("--cuda", default="data/phase2/results_cuda.csv")
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--tex-dir", required=True)
    args = ap.parse_args(argv)
    build(args.cpu, args.cuda, args.fig_dir, args.tex_dir)


if __name__ == "__main__":
    main()
