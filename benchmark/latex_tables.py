from __future__ import annotations

from benchmark.combine import models_for_task, value_for
from benchmark.measure import latency_stats

_RES = 640


def _fmt(v, ndigits=0):
    if v is None:
        return "--"
    return f"{v:.{ndigits}f}"


def latency_table(rows, task, label) -> str:
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Measured warm inference latency (this work) at {_RES}px for "
        rf"{task} models: CPU vs.\ GPU, with CPU throughput.}}",
        rf"\label{{{label}}}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{}Xrrr@{}}", r"\toprule",
        r"Model & CPU latency (ms) & GPU latency (ms) & CPU throughput (img/s) \\", r"\midrule",
    ]
    for model in models_for_task(rows, task):
        cpu = value_for(rows, device="cpu", model=model, experiment="warm_latency",
                        metric="mean_ms", resolution=_RES)
        gpu = value_for(rows, device="cuda", model=model, experiment="warm_latency",
                        metric="mean_ms", resolution=_RES)
        thr = value_for(rows, device="cpu", model=model, experiment="throughput",
                        metric="imgs_per_sec", resolution=_RES)
        lines.append(rf"{model} & {_fmt(cpu)} & {_fmt(gpu)} & {_fmt(thr, 2)} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def coldwarm_table(rows, task, label) -> str:
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{First-inference cold start (lazy graph build and on-demand downloads; the "
        rf"model object is already constructed) vs.\ warm median latency at {_RES}px for "
        rf"{task} models (this work, CPU).}}",
        rf"\label{{{label}}}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{}Xrr@{}}", r"\toprule",
        r"Model & Cold-start (ms) & Warm median (ms) \\", r"\midrule",
    ]
    for model in models_for_task(rows, task):
        cold = value_for(rows, device="cpu", model=model, experiment="cold_start",
                         metric="first_call_ms", resolution=_RES)
        warm = value_for(rows, device="cpu", model=model, experiment="warm_latency",
                         metric="median_ms", resolution=_RES)
        lines.append(rf"{model} & {_fmt(cold)} & {_fmt(warm)} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def memory_table(rows, label) -> str:
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Peak memory at {_RES}px (this work): host RSS on CPU, CUDA VRAM on GPU.}}",
        rf"\label{{{label}}}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{}Xrr@{}}", r"\toprule",
        r"Model & CPU peak RSS (MB) & GPU peak VRAM (MB) \\", r"\midrule",
    ]
    models = sorted({r["model"] for r in rows})
    for model in models:
        rss = value_for(rows, device="cpu", model=model, experiment="peak_rss",
                        metric="rss_mb", resolution=_RES)
        vram = value_for(rows, device="cuda", model=model, experiment="peak_gpu",
                         metric="gpu_mem_mb", resolution=_RES)
        lines.append(rf"{model} & {_fmt(rss)} & {_fmt(vram)} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def distribution_table(dist_rows, label) -> str:
    devices = [d for d in ("cpu", "cuda") if any(r["device"] == d for r in dist_rows)]
    models = sorted({r["model"] for r in dist_rows})
    n = max((len([r for r in dist_rows if r["device"] == devices[0]
                  and r["model"] == m]) for m in models), default=0) if devices else 0
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Warm inference latency distribution at {_RES}px over {n} images "
        rf"(this work), one timed inference per image after warmup: mean, std, and "
        rf"P50/P90/P99 percentiles.}}",
        rf"\label{{{label}}}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{}Xlrrrrr@{}}", r"\toprule",
        r"Model & Device & Mean (ms) & Std (ms) & P50 (ms) & P90 (ms) & P99 (ms) \\",
        r"\midrule",
    ]
    for model in models:
        for device in devices:
            samples = [r["latency_ms"] for r in dist_rows
                       if r["model"] == model and r["device"] == device]
            if not samples:
                continue
            s = latency_stats(samples)
            lines.append(
                rf"{model} & {device} & {_fmt(s['mean_ms'], 1)} & {_fmt(s['std_ms'], 1)} & "
                rf"{_fmt(s['p50_ms'], 1)} & {_fmt(s['p90_ms'], 1)} & {_fmt(s['p99_ms'], 1)} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def feasibility_table(rows, label) -> str:
    lines = [
        r"\begin{table*}[ht]", r"\centering",
        rf"\caption{{Deployment feasibility on the $\sim$1~GB tier: isolated peak host RSS "
        rf"at {_RES}px (this work) and whether the model fits under 1024~MB.}}",
        rf"\label{{{label}}}", r"\small",
        r"\begin{tabularx}{\linewidth}{@{}Xrr@{}}", r"\toprule",
        r"Model & Peak RSS (MB) & Fits $<$1~GB \\", r"\midrule",
    ]
    for model in sorted({r["model"] for r in rows}):
        rss = value_for(rows, device="cpu", model=model, experiment="peak_rss",
                        metric="rss_mb", resolution=_RES)
        fits = "--" if rss is None else ("Yes" if rss < 1024 else "No")
        lines.append(rf"{model} & {_fmt(rss)} & {fits} \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"
