"""Download a fixed set of COCO val2017 images for the performance campaign.

Performance scales with pixel count, not labels, so any valid val2017 images work;
these eight low IDs are stable, well-known entries of the official val2017 split.
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import urllib.request
import zipfile

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


# --- Distribution campaign (Phase 2c): a fixed ~100-image set for latency percentiles. ---
# The coco128 subset is 128 real COCO images in one pinned zip; latency depends on pixel
# content, not labels, so this is a reproducible stand-in for "100 varied real images".
DIST_ZIP_URL = "https://ultralytics.com/assets/coco128.zip"


def dist_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "dist_images"


def _extract_jpgs(zip_path, out_dir, limit) -> list[pathlib.Path]:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collected = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))
        for name in names[:limit]:
            dest = out_dir / pathlib.Path(name).name
            dest.write_bytes(zf.read(name))
            collected.append(dest)
    return sorted(collected)


def fetch_dist_images(limit: int = 100) -> list[pathlib.Path]:
    out = dist_dir()
    existing = sorted(out.glob("*.jpg")) if out.exists() else []
    if len(existing) >= limit:
        return existing[:limit]
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    try:
        urllib.request.urlretrieve(DIST_ZIP_URL, tmp.name)
        return _extract_jpgs(tmp.name, out, limit)
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    for p in fetch_all():
        print(p)
