from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def load_rows(path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["resolution"] = int(r["resolution"])
            r["value"] = float(r["value"])
            rows.append(r)
    return rows


def _filter(rows, **eq):
    return [r for r in rows if all(r[k] == v for k, v in eq.items())]


def _legend(ax, **kw):
    # Only draw a legend when labeled artists exist, to avoid a UserWarning on empty data.
    if ax.get_legend_handles_labels()[1]:
        ax.legend(**kw)


def plot_cpu_vs_gpu(rows, out_path):
    sel = _filter(rows, experiment="warm_latency", metric="mean_ms", resolution=640)
    models = sorted({r["model"] for r in sel})
    fig, ax = plt.subplots()
    for i, device in enumerate(["cpu", "cuda"]):
        vals = [next((r["value"] for r in sel if r["model"] == m and r["device"] == device), 0.0)
                for m in models]
        ax.bar([x + i * 0.4 for x in range(len(models))], vals, width=0.4, label=device)
    ax.set_xticks([x + 0.2 for x in range(len(models))])
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("warm latency (ms), res 640")
    # Log scale so the small GPU bars stay visible next to the much larger CPU bars.
    if sel:
        ax.set_yscale("log")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scaling(rows, out_path):
    sel = _filter(rows, experiment="warm_latency", metric="mean_ms")
    fig, ax = plt.subplots()
    # Separate per (model, device): mixing CPU and GPU into one line per model produces a
    # meaningless sawtooth. Log y because CPU and GPU latencies span two orders of magnitude.
    for device in sorted({r["device"] for r in sel}):
        for model in sorted({r["model"] for r in sel if r["device"] == device}):
            pts = sorted((r["resolution"], r["value"]) for r in sel
                         if r["model"] == model and r["device"] == device)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", label=f"{model} ({device})")
    ax.set_xlabel("resolution (px)")
    ax.set_ylabel("warm latency (ms)")
    if sel:
        ax.set_yscale("log")
    _legend(ax, fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_memory_scaling(rows, out_path):
    mem = [r for r in rows if r["experiment"] in ("peak_rss", "peak_gpu")]
    fig, ax = plt.subplots()
    for device in sorted({r["device"] for r in mem}):
        for model in sorted({r["model"] for r in mem if r["device"] == device}):
            pts = sorted((r["resolution"], r["value"]) for r in mem
                         if r["device"] == device and r["model"] == model)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", label=f"{model} ({device})")
    ax.set_xlabel("resolution (px)")
    ax.set_ylabel("peak memory (MB)")
    _legend(ax, fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
