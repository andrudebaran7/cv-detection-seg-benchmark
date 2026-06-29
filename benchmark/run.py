from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib

from benchmark.images import RESOLUTIONS, load_images, resize
from benchmark.manifest import build_manifest
from benchmark.measure import (
    peak_gpu_mb,
    peak_rss_mb,
    reset_peak_gpu,
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


def run_model(spec, image, *, device, resolution, iters, warmup):
    img = resize(image, resolution)
    rows = []

    # B2 cold-start: a fresh model instance, first call only.
    cold_model = spec.factory(device)
    cold_ms = time_first_call_ms(
        lambda: cold_model.predict(img, **spec.predict_kwargs(img))
    )
    rows.append(_row(spec, device, resolution, "cold_start", "first_call_ms", cold_ms, 1))
    del cold_model  # free the cold instance so the memory reading isolates the warm model

    # Warm model reused for B1/B3/B5.
    model = spec.factory(device)
    kwargs = spec.predict_kwargs(img)
    call = lambda: model.predict(img, **kwargs)

    # B5 memory: reset CUDA peak tracking so the warm run's peak is isolated (no-op on CPU).
    reset_peak_gpu()
    stats = timeit_callable(call, warmup=warmup, iters=iters)
    rows.append(_row(spec, device, resolution, "warm_latency", "mean_ms", stats["mean_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "median_ms", stats["median_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "std_ms", stats["std_ms"], iters))

    throughput = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
    rows.append(_row(spec, device, resolution, "throughput", "imgs_per_sec", throughput, iters))

    # Device-appropriate memory: host RSS on CPU, peak CUDA VRAM on GPU (spec B5).
    if device == "cuda":
        rows.append(_row(spec, device, resolution, "peak_gpu", "gpu_mem_mb", peak_gpu_mb(), 1))
    else:
        rows.append(_row(spec, device, resolution, "peak_rss", "rss_mb", peak_rss_mb(), 1))
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
