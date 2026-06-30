from __future__ import annotations

from benchmark.combine import models_for_task, value_for

_RES = 640


def _fmt(v, ndigits=0):
    if v is None:
        return "--"
    return f"{v:.{ndigits}f}"


def latency_table(rows, task, label) -> str:
    lines = [
        r"\begin{table}[ht]", r"\centering",
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
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def coldwarm_table(rows, task, label) -> str:
    lines = [
        r"\begin{table}[ht]", r"\centering",
        rf"\caption{{Cold-start (first call, incl.\ load) vs.\ warm median latency at {_RES}px "
        rf"for {task} models (this work, CPU).}}",
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
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def memory_table(rows, label) -> str:
    lines = [
        r"\begin{table}[ht]", r"\centering",
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
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines) + "\n"
