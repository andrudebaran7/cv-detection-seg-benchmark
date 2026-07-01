from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import subprocess
import sys

from benchmark.images import RESOLUTIONS, load_images, resize
from benchmark.manifest import build_manifest
from benchmark.measure import (
    time_first_call_ms,
    timeit_callable,
)
from benchmark.models_registry import REGISTRY

_COLUMNS = ["device", "model", "task", "image", "resolution",
            "experiment", "metric", "value", "n_iters", "measured_at"]


def _row(spec, device, resolution, experiment, metric, value, n_iters):
    return {
        "device": device, "model": spec.key, "task": spec.task,
        "image": "set", "resolution": resolution, "experiment": experiment,
        "metric": metric, "value": float(value), "n_iters": n_iters,
        "measured_at": datetime.date.today().isoformat(),
    }


def _probe_memory(model_key, device, resolution):
    """Measure peak memory for one model in a FRESH subprocess (true per-model isolation)."""
    from benchmark.mem_probe import RESULT_MARKER

    proc = subprocess.run(
        [sys.executable, "-m", "benchmark.mem_probe",
         "--model", model_key, "--device", device, "--resolution", str(resolution)],
        capture_output=True, text=True, check=True,
    )
    # Find the sentinel-marked result line; do NOT trust "the last line" (verbose model
    # libraries print around it, which previously mis-parsed rfdetr/mask2former).
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise RuntimeError(f"mem_probe produced no result line for {model_key} @ {resolution}")


def run_model(spec, image, *, device, resolution, iters, warmup):
    img = resize(image, resolution)
    rows = []

    # B2 cold-start: a fresh model instance, first call only. The model object is already
    # constructed (weights loaded in the constructor for ultralytics/transformers); this
    # times the first inference (lazy graph build + any on-demand downloads).
    cold_model = spec.factory(device)
    cold_ms = time_first_call_ms(
        lambda: cold_model.predict(img, **spec.predict_kwargs(img))
    )
    rows.append(_row(spec, device, resolution, "cold_start", "first_call_ms", cold_ms, 1))
    del cold_model

    # Warm model reused for B1/B3.
    model = spec.factory(device)
    kwargs = spec.predict_kwargs(img)
    call = lambda: model.predict(img, **kwargs)

    stats = timeit_callable(call, warmup=warmup, iters=iters)
    rows.append(_row(spec, device, resolution, "warm_latency", "mean_ms", stats["mean_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "median_ms", stats["median_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "std_ms", stats["std_ms"], iters))

    throughput = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
    rows.append(_row(spec, device, resolution, "throughput", "imgs_per_sec", throughput, iters))
    return rows


def memory_rows(spec, *, device, resolutions):
    """Memory rows for one model across resolutions, via the isolated subprocess probe.

    Call this BEFORE any model is loaded in-process: the probe forks from the current
    process, and ``ru_maxrss`` in the child inherits the parent's fork-time RSS, so a heavy
    parent (one that already loaded models for latency timing) would inflate the reading.
    """
    rows = []
    for resolution in resolutions:
        mem = _probe_memory(spec.key, device, resolution)
        if device == "cuda":
            rows.append(_row(spec, device, resolution, "peak_gpu", "gpu_mem_mb", mem["gpu_mb"], 1))
        else:
            rows.append(_row(spec, device, resolution, "peak_rss", "rss_mb", mem["rss_mb"], 1))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2 performance campaign")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--models", default="all")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="data/phase2")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(REGISTRY) if args.models == "all" else args.models.split(",")
    base = load_images()[0]  # resolution sweep uses one base image (B4)
    failed = {}

    # Phase A --- MEMORY FIRST, while this process is still light (no model loaded in it yet),
    # so each isolated subprocess probe reports only its own model (see memory_rows()).
    mem_by_model = {}
    for key in keys:
        try:
            mem_by_model[key] = memory_rows(REGISTRY[key], device=args.device, resolutions=RESOLUTIONS)
        except Exception as exc:
            failed[key] = repr(exc)
            print(f"[skip] {key} (memory): {exc}")

    # Phase B --- latency/throughput/cold-start in-process (this makes the process heavy, but
    # memory is already captured). Only models that passed Phase A are attempted.
    lat_by_model = {}
    for key in [k for k in keys if k not in failed]:
        try:
            lat_by_model[key] = [
                r for resolution in RESOLUTIONS
                for r in run_model(REGISTRY[key], base, device=args.device,
                                   resolution=resolution, iters=args.iters, warmup=args.warmup)
            ]
        except Exception as exc:
            failed[key] = repr(exc)
            print(f"[skip] {key} (latency): {exc}")

    # Emit only models that survived BOTH phases (skipped == no data).
    rows = []
    for key in keys:
        if key in failed:
            continue
        rows.extend(lat_by_model.get(key, []))
        rows.extend(mem_by_model.get(key, []))

    write_csv(rows, out / f"results_{args.device}.csv")
    with open(out / f"manifest_{args.device}.json", "w") as f:
        json.dump(build_manifest(args.device), f, indent=2)
        f.write("\n")
    if failed:
        print(f"Completed with {len(failed)} model(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
