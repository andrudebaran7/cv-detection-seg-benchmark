"""Per-model peak-memory probe, run as a fresh subprocess for isolation.

Loading only one model in a clean process is what makes the reading attributable to
that model: the CPU figure is this process's true peak RSS
(``resource.getrusage(...).ru_maxrss``, a real high-water mark, not an instantaneous
sample), and the GPU figure is ``torch.cuda.max_memory_allocated`` since reset. This
avoids the in-process pitfalls the paper itself documents: dropping a reference plus
``gc`` does not lower RSS without ``malloc_trim``, so measuring several models in one
process double-counts retained pages.

Usage: ``python -m benchmark.mem_probe --model <key> --device {cpu,cuda} --resolution N``
prints one JSON line ``{"rss_mb": ..., "gpu_mb": ...}``.
"""
from __future__ import annotations

import argparse
import json
import resource

from benchmark.images import load_images, resize
from benchmark.measure import peak_gpu_mb, reset_peak_gpu
from benchmark.models_registry import REGISTRY


def measure_peak_memory(model_key, device, resolution, *, iters=5, warmup=2) -> dict:
    spec = REGISTRY[model_key]
    img = resize(load_images()[0], resolution)
    model = spec.factory(device)
    kwargs = spec.predict_kwargs(img)
    reset_peak_gpu()
    for _ in range(warmup + iters):
        model.predict(img, **kwargs)
    # ru_maxrss is in kilobytes on Linux -> MB.
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"rss_mb": rss_mb, "gpu_mb": peak_gpu_mb()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-model peak-memory probe (run isolated)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args(argv)
    out = measure_peak_memory(args.model, args.device, args.resolution,
                              iters=args.iters, warmup=args.warmup)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
