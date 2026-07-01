from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib

from PIL import Image

from benchmark import fetch_images
from benchmark.images import resize
from benchmark.manifest import build_manifest
from benchmark.measure import time_per_image
from benchmark.models_registry import REGISTRY

RES = 640

_DIST_COLUMNS = ["device", "model", "task", "image_id",
                 "latency_ms", "resolution", "measured_at"]


def _dist_row(spec, device, image_id, latency_ms):
    return {
        "device": device, "model": spec.key, "task": spec.task,
        "image_id": image_id, "latency_ms": float(latency_ms),
        "resolution": RES, "measured_at": datetime.date.today().isoformat(),
    }


def run_dist_model(spec, images, image_ids, *, device, warmup=5) -> list[dict]:
    """Time one inference per image for one model (loaded once), at RES px."""
    model = spec.factory(device)
    resized = [resize(im, RES) for im in images]

    def predict_one(img):
        return model.predict(img, **spec.predict_kwargs(img))

    samples = time_per_image(predict_one, resized, warmup=warmup)
    return [_dist_row(spec, device, image_id, ms)
            for image_id, ms in zip(image_ids, samples)]


def write_dist_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DIST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2c latency-distribution campaign")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--models", default="all")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="data/phase2")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = fetch_images.fetch_dist_images(args.limit)
    images = [Image.open(p).convert("RGB") for p in paths]
    image_ids = [p.stem for p in paths]

    keys = list(REGISTRY) if args.models == "all" else args.models.split(",")
    rows = []
    failed = {}
    for key in keys:
        try:
            rows.extend(run_dist_model(REGISTRY[key], images, image_ids,
                                       device=args.device, warmup=args.warmup))
        except Exception as exc:  # a failing model is skipped; others still write
            failed[key] = repr(exc)
            print(f"[skip] {key}: {exc}")

    write_dist_csv(rows, out / f"results_dist_{args.device}.csv")
    with open(out / f"manifest_dist_{args.device}.json", "w") as f:
        json.dump(build_manifest(args.device), f, indent=2)
        f.write("\n")
    if failed:
        print(f"Completed with {len(failed)} model(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
