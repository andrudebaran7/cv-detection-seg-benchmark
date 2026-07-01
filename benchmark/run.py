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

    # B5 memory: measured in an isolated subprocess (peak RSS on CPU, peak CUDA VRAM on GPU).
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
    imgs = load_images()
    base = imgs[0]  # resolution sweep uses one base image (B4)

    rows = []
    failed = {}
    for key in keys:
        spec = REGISTRY[key]
        mark = len(rows)  # roll back to here if this model fails mid-sweep
        try:
            for resolution in RESOLUTIONS:
                rows.extend(run_model(spec, base, device=args.device,
                                      resolution=resolution, iters=args.iters, warmup=args.warmup))
        except Exception as exc:  # one model's failure must not abort the whole campaign
            del rows[mark:]  # drop any partial rows so a skipped model leaves none
            failed[key] = repr(exc)
            print(f"[skip] {key}: {exc}")

    write_csv(rows, out / f"results_{args.device}.csv")
    with open(out / f"manifest_{args.device}.json", "w") as f:
        json.dump(build_manifest(args.device), f, indent=2)
        f.write("\n")
    if failed:
        print(f"Completed with {len(failed)} model(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
