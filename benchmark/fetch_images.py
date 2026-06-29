"""Download a fixed set of COCO val2017 images for the performance campaign.

Performance scales with pixel count, not labels, so any valid val2017 images work;
these eight low IDs are stable, well-known entries of the official val2017 split.
"""
from __future__ import annotations

import pathlib
import urllib.request

# Stable val2017 image IDs (the official COCO image URL pattern is permanent).
IMAGE_IDS = [139, 285, 632, 724, 776, 785, 802, 872]


def url_for(image_id: int) -> str:
    return f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg"


def target_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "benchmark_images"


def fetch_all() -> list[pathlib.Path]:
    out = target_dir()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for image_id in IMAGE_IDS:
        dest = out / f"{image_id:012d}.jpg"
        if not dest.exists():
            urllib.request.urlretrieve(url_for(image_id), dest)
        paths.append(dest)
    return paths


if __name__ == "__main__":
    for p in fetch_all():
        print(p)
