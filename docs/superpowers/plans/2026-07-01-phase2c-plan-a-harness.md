# Phase 2c — Plan A: Latency-Distribution Harness (benchmark repo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a latency-distribution campaign that times each of ~100 real images once per model, computes P50/P90/P99, and derives a 1 GB deployment-feasibility table — producing the CSV, table, and box-plot the paper (Plan B) will consume.

**Architecture:** A new campaign module (`benchmark/run_dist.py`) loads each model once, times a single inference per image over a fixed 100-image set, and writes raw per-image samples to `data/phase2/results_dist_{device}.csv`. Two small pure primitives in `measure.py` (`latency_stats`, `time_per_image`) do the statistics and timing; a loader in `combine.py` reads the distribution CSV; generators in `latex_tables.py` and `plot.py` turn the samples into a LaTeX table + box-plot, and a feasibility table is derived from the existing isolated peak-RSS numbers. This plan ends with the **CPU** run; the author then runs the same command on Colab for the **GPU** CSV.

**Tech Stack:** Python 3.9+, `statistics` (stdlib), `matplotlib` (Agg), `PIL`, `pytest`. No new dependencies.

## Global Constraints

- **Image source:** the 100-image set is the **`coco128`** subset (128 real COCO images, one pinned zip `https://ultralytics.com/assets/coco128.zip`), NOT 100 individually-enumerated val2017 ids. Reason: no accuracy is measured (only inference latency over varied content), and one pinned zip is far more reproducible than a fragile 100-id list. Images live in `data/dist_images/` (git-ignored); reproducibility comes from the pinned zip URL.
- **Distribution resolution:** fixed at **640 px** (`RES = 640`), matching the existing per-resolution campaign's reporting point.
- **Sampling protocol:** load model once → global warmup → **one timed inference per image** → 100 samples per model×device. The distribution is over image content.
- **CSV schema (distribution):** columns exactly `device, model, task, image_id, latency_ms, resolution, measured_at`. Separate file from the existing `results_{device}.csv`; never mix schemas.
- **Statistics:** mean (`statistics.fmean`), std (`statistics.pstdev`, population — matches `timeit_callable`), P50/P90/P99 via linear-interpolation percentile.
- **1 GB feasibility rule:** `fits` iff isolated `peak_rss` at 640 `< 1024` MB, read from the existing `data/phase2/results_cpu.csv`.
- **Resilience:** per-model `try/except` mirrors `run.py` — a failing model is skipped (printed `[skip]`), others still write.
- **Tests:** TDD, network-free (build zip fixtures in-test; stub models/fetch), consistent with the existing `tests/test_*.py` style. Keep the suite green.
- **No new runtime deps.** Commit frequently (one commit per task minimum).

---

### Task 1: coco128 distribution-image fetch

**Files:**
- Modify: `benchmark/fetch_images.py`
- Modify: `.gitignore`
- Test: `tests/test_fetch_images.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `fetch_images.DIST_ZIP_URL: str`
  - `fetch_images.dist_dir() -> pathlib.Path` (repo `data/dist_images`)
  - `fetch_images._extract_jpgs(zip_path, out_dir, limit) -> list[pathlib.Path]` (sorted, ≤ limit)
  - `fetch_images.fetch_dist_images(limit: int = 100) -> list[pathlib.Path]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fetch_images.py`:

```python
import io
import zipfile

from benchmark import fetch_images


def test_dist_dir_points_at_repo_data():
    p = fetch_images.dist_dir()
    assert p.name == "dist_images"
    assert p.parent.name == "data"


def test_dist_zip_url_is_coco128():
    assert fetch_images.DIST_ZIP_URL.endswith("coco128.zip")


def _make_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            # a tiny valid-enough byte payload; extraction copies bytes verbatim
            zf.writestr(name, b"\xff\xd8\xff\xe0jpegbytes")


def test_extract_jpgs_collects_sorted_and_limited(tmp_path):
    zpath = tmp_path / "coco128.zip"
    _make_zip(zpath, [
        "coco128/images/train2017/000000000009.jpg",
        "coco128/images/train2017/000000000025.jpg",
        "coco128/images/train2017/000000000030.jpg",
        "coco128/labels/train2017/000000000009.txt",  # non-jpg, must be ignored
    ])
    out = tmp_path / "dist_images"
    got = fetch_images._extract_jpgs(zpath, out, limit=2)
    assert [p.name for p in got] == ["000000000009.jpg", "000000000025.jpg"]
    assert all(p.exists() for p in got)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fetch_images.py -v`
Expected: FAIL (`AttributeError: module 'benchmark.fetch_images' has no attribute 'dist_dir'`)

- [ ] **Step 3: Implement the fetch helpers**

Append to `benchmark/fetch_images.py` (keep existing `IMAGE_IDS`, `url_for`, `target_dir`, `fetch_all`); add imports at the top of the file next to the existing ones:

```python
import os
import tempfile
import zipfile
```

Then append:

```python
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
```

- [ ] **Step 4: Add `data/dist_images/` to `.gitignore`**

Add this line to `.gitignore` (after `data/phase2/*.png`):

```
data/dist_images/
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fetch_images.py -v`
Expected: PASS (all tests, including the three pre-existing ones)

- [ ] **Step 6: Commit**

```bash
git add benchmark/fetch_images.py tests/test_fetch_images.py .gitignore
git commit -m "feat(bench): add coco128 distribution-image fetch (Phase 2c)"
```

---

### Task 2: measurement primitives — `latency_stats` + `time_per_image`

**Files:**
- Modify: `benchmark/measure.py`
- Test: `tests/test_measure.py`

**Interfaces:**
- Consumes: existing `measure._cuda_sync_fn()` (module-internal, resolves a CUDA barrier once).
- Produces:
  - `measure._percentile(sorted_samples: list[float], pct: float) -> float` (linear interpolation, numpy-style)
  - `measure.latency_stats(samples_ms: list[float]) -> dict` with keys `n, mean_ms, std_ms, p50_ms, p90_ms, p99_ms`
  - `measure.time_per_image(predict_one, images, *, warmup: int = 5) -> list[float]` — one timed call per image (ms), after `warmup` warmup calls on `images[0]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_measure.py`:

```python
def test_percentile_linear_interpolation():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert measure._percentile(data, 50) == 55.0
    assert measure._percentile(data, 90) == 91.0
    assert round(measure._percentile(data, 99), 4) == 99.1


def test_latency_stats_reports_mean_std_percentiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    s = measure.latency_stats(data)
    assert s["n"] == 10
    assert s["mean_ms"] == 55.0
    assert s["p50_ms"] == 55.0
    assert s["p90_ms"] == 91.0
    assert s["std_ms"] >= 0.0


def test_latency_stats_single_sample():
    s = measure.latency_stats([42.0])
    assert s["n"] == 1
    assert s["mean_ms"] == 42.0
    assert s["p50_ms"] == 42.0 and s["p90_ms"] == 42.0 and s["p99_ms"] == 42.0
    assert s["std_ms"] == 0.0


def test_time_per_image_one_sample_per_image_after_warmup():
    calls = {"n": 0}

    def predict_one(img):
        calls["n"] += 1

    images = ["a", "b", "c"]
    samples = measure.time_per_image(predict_one, images, warmup=2)
    assert len(samples) == 3
    assert calls["n"] == 2 + 3  # warmup on images[0] + one per image
    assert all(ms >= 0.0 for ms in samples)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_measure.py -v`
Expected: FAIL (`AttributeError: module 'benchmark.measure' has no attribute '_percentile'`)

- [ ] **Step 3: Implement the primitives**

Append to `benchmark/measure.py`:

```python
def _percentile(sorted_samples, pct: float) -> float:
    """Linear-interpolation percentile (numpy default), pct in [0, 100]."""
    if not sorted_samples:
        raise ValueError("empty samples")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (len(sorted_samples) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return float(sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac)


def latency_stats(samples_ms) -> dict:
    """Distribution summary over per-image latency samples (ms)."""
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "n": n,
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered) if n > 1 else 0.0,
        "p50_ms": _percentile(ordered, 50),
        "p90_ms": _percentile(ordered, 90),
        "p99_ms": _percentile(ordered, 99),
    }


def time_per_image(predict_one: Callable[[object], object], images, *, warmup: int = 5) -> list:
    """Time one inference per image (ms) after a global warmup on the first image.

    ``predict_one(img)`` runs one inference. CUDA work is drained per call so wall-clock
    timing attributes the full GPU cost (mirrors ``timeit_callable``).
    """
    sync = _cuda_sync_fn()
    if images:
        for _ in range(warmup):
            predict_one(images[0])
        sync()
    samples_ms = []
    for img in images:
        start = time.perf_counter()
        predict_one(img)
        sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_measure.py -v`
Expected: PASS (new tests + the three pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add benchmark/measure.py tests/test_measure.py
git commit -m "feat(bench): add latency_stats + time_per_image primitives (Phase 2c)"
```

---

### Task 3: distribution CSV loader in `combine.py`

**Files:**
- Modify: `benchmark/combine.py`
- Test: `tests/test_combine.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `combine.load_dist(*paths) -> list[dict]` — reads distribution CSVs, coercing `latency_ms` to `float` and `resolution` to `int`; leaves `device, model, task, image_id, measured_at` as strings.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_combine.py`:

```python
_DIST_COLS = ["device", "model", "task", "image_id", "latency_ms", "resolution", "measured_at"]


def _write_dist(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_DIST_COLS)
        w.writerows(rows)


def test_load_dist_coerces_latency_and_resolution(tmp_path):
    _write_dist(tmp_path / "d.csv", [
        ["cpu", "yolo11n", "detection", "000000000009", 201.5, 640, "2026-07-01"],
        ["cpu", "yolo11n", "detection", "000000000025", 198.0, 640, "2026-07-01"]])
    rows = combine.load_dist(tmp_path / "d.csv")
    assert len(rows) == 2
    assert rows[0]["latency_ms"] == 201.5 and isinstance(rows[0]["latency_ms"], float)
    assert rows[0]["resolution"] == 640 and isinstance(rows[0]["resolution"], int)
    assert rows[0]["image_id"] == "000000000009"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_combine.py -v`
Expected: FAIL (`AttributeError: module 'benchmark.combine' has no attribute 'load_dist'`)

- [ ] **Step 3: Implement the loader**

Append to `benchmark/combine.py`:

```python
def load_dist(*paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                r["latency_ms"] = float(r["latency_ms"])
                r["resolution"] = int(r["resolution"])
                rows.append(r)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_combine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add benchmark/combine.py tests/test_combine.py
git commit -m "feat(bench): add load_dist for distribution CSVs (Phase 2c)"
```

---

### Task 4: distribution campaign `run_dist.py`

**Files:**
- Create: `benchmark/run_dist.py`
- Test: `tests/test_run_dist.py`

**Interfaces:**
- Consumes: `benchmark.images.resize`, `benchmark.measure.time_per_image`, `benchmark.models_registry.REGISTRY` and `ModelSpec` (`.key`, `.task`, `.factory(device)`, `.predict_kwargs(img)`), `benchmark.fetch_images.fetch_dist_images`, `benchmark.manifest.build_manifest`.
- Produces:
  - `run_dist.RES = 640`
  - `run_dist._DIST_COLUMNS` (list, exact schema)
  - `run_dist.run_dist_model(spec, images, image_ids, *, device, warmup=5) -> list[dict]`
  - `run_dist.write_dist_csv(rows, path)`
  - `run_dist.main(argv=None)` — CLI writing `results_dist_{device}.csv` + `manifest_dist_{device}.json`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_dist.py`:

```python
import csv

from PIL import Image

from benchmark import run_dist
from benchmark.models_registry import ModelSpec


class _FakeModel:
    def __init__(self, device=None):
        self.device = device

    def predict(self, image, **kwargs):
        class P:
            latency_ms = 1.0
        return P()


def _fake_spec():
    return ModelSpec("fake", "detection", lambda device: _FakeModel(device), lambda img: {})


def test_run_dist_model_one_row_per_image():
    images = [Image.new("RGB", (100, 80)) for _ in range(4)]
    ids = ["a", "b", "c", "d"]
    rows = run_dist.run_dist_model(_fake_spec(), images, ids, device="cpu", warmup=1)
    assert len(rows) == 4
    assert [r["image_id"] for r in rows] == ids
    for r in rows:
        assert r["device"] == "cpu" and r["model"] == "fake" and r["task"] == "detection"
        assert r["resolution"] == 640
        assert isinstance(r["latency_ms"], float)


def test_write_dist_csv_has_fixed_header(tmp_path):
    rows = [{"device": "cpu", "model": "fake", "task": "detection", "image_id": "a",
             "latency_ms": 1.0, "resolution": 640, "measured_at": "2026-07-01"}]
    p = tmp_path / "results_dist_cpu.csv"
    run_dist.write_dist_csv(rows, p)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["device", "model", "task", "image_id",
                      "latency_ms", "resolution", "measured_at"]


def test_main_writes_csv_and_manifest(tmp_path, monkeypatch):
    imgs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    paths = []
    for i, im in enumerate(imgs):
        pp = tmp_path / f"{i:012d}.jpg"
        im.save(pp)
        paths.append(pp)
    monkeypatch.setattr(run_dist.fetch_images, "fetch_dist_images", lambda limit=100: paths)
    monkeypatch.setattr(run_dist, "REGISTRY", {"fake": _fake_spec()})
    out = tmp_path / "phase2"
    run_dist.main(["--device", "cpu", "--out", str(out), "--limit", "3", "--warmup", "1"])
    assert (out / "results_dist_cpu.csv").exists()
    assert (out / "manifest_dist_cpu.json").exists()
    with open(out / "results_dist_cpu.csv") as f:
        data = list(csv.DictReader(f))
    assert len(data) == 3 and all(d["model"] == "fake" for d in data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_run_dist.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'benchmark.run_dist'`)

- [ ] **Step 3: Implement `run_dist.py`**

Create `benchmark/run_dist.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_run_dist.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add benchmark/run_dist.py tests/test_run_dist.py
git commit -m "feat(bench): add latency-distribution campaign run_dist (Phase 2c)"
```

---

### Task 5: LaTeX generators — `distribution_table` + `feasibility_table`

**Files:**
- Modify: `benchmark/latex_tables.py`
- Test: `tests/test_latex_tables.py`

**Interfaces:**
- Consumes: `measure.latency_stats` (Task 2); existing `combine.value_for`; existing `_fmt`, `_RES`.
- Produces:
  - `latex_tables.distribution_table(dist_rows, label) -> str` — one row per model×device: mean, std, P50, P90, P99 (ms) at 640, computed from `latency_ms` samples.
  - `latex_tables.feasibility_table(rows, label) -> str` — one row per model: isolated CPU peak RSS (MB) and `Yes`/`No` fits on the ~1 GB tier (`< 1024` MB).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_latex_tables.py`:

```python
_DIST_ROWS = [
    {"device": "cpu", "model": "yolo11n", "task": "detection",
     "latency_ms": 200.0, "resolution": 640},
    {"device": "cpu", "model": "yolo11n", "task": "detection",
     "latency_ms": 220.0, "resolution": 640},
    {"device": "cuda", "model": "yolo11n", "task": "detection",
     "latency_ms": 9.0, "resolution": 640},
    {"device": "cuda", "model": "yolo11n", "task": "detection",
     "latency_ms": 11.0, "resolution": 640},
]

_MEM_ROWS = [
    {"device": "cpu", "model": "yolo11n", "task": "detection", "resolution": 640,
     "experiment": "peak_rss", "metric": "rss_mb", "value": 468.0},
    {"device": "cpu", "model": "sam2-tiny", "task": "segmentation", "resolution": 640,
     "experiment": "peak_rss", "metric": "rss_mb", "value": 1600.0},
]


def test_distribution_table_has_percentiles_and_devices():
    tex = lt.distribution_table(_DIST_ROWS, "tab:latency-dist")
    assert "\\label{tab:latency-dist}" in tex
    assert "yolo11n" in tex
    assert "cpu" in tex and "cuda" in tex
    assert "P50" in tex and "P90" in tex and "P99" in tex
    assert "\\begin{tabularx}" in tex and "\\toprule" in tex


def test_feasibility_table_marks_fit_and_oom():
    tex = lt.feasibility_table(_MEM_ROWS, "tab:feasibility")
    assert "\\label{tab:feasibility}" in tex
    assert "468" in tex and "Yes" in tex        # yolo11n fits
    assert "1600" in tex and "No" in tex        # sam2-tiny OOM
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_latex_tables.py -v`
Expected: FAIL (`AttributeError: module 'benchmark.latex_tables' has no attribute 'distribution_table'`)

- [ ] **Step 3: Implement the generators**

At the top of `benchmark/latex_tables.py`, extend the imports:

```python
from benchmark.combine import models_for_task, value_for
from benchmark.measure import latency_stats
```

Append to `benchmark/latex_tables.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_latex_tables.py -v`
Expected: PASS (new tests + the two pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add benchmark/latex_tables.py tests/test_latex_tables.py
git commit -m "feat(bench): add distribution + 1GB feasibility LaTeX tables (Phase 2c)"
```

---

### Task 6: box-plot generator `plot_latency_boxplot`

**Files:**
- Modify: `benchmark/plot.py`
- Test: `tests/test_plot.py`

**Interfaces:**
- Consumes: `matplotlib` (already imported in `plot.py`).
- Produces: `plot.plot_latency_boxplot(rows, out_path)` — a box plot of per-image `latency_ms` samples grouped per model×device (log y), saved to `out_path`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_plot.py`:

```python
def test_plot_latency_boxplot_writes_file(tmp_path):
    rows = []
    for dev, base in (("cpu", 200.0), ("cuda", 9.0)):
        for i in range(10):
            rows.append({"device": dev, "model": "yolo11n", "task": "detection",
                         "latency_ms": base + i, "resolution": 640})
    out = tmp_path / "box.png"
    plot.plot_latency_boxplot(rows, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plot.py -v`
Expected: FAIL (`AttributeError: module 'benchmark.plot' has no attribute 'plot_latency_boxplot'`)

- [ ] **Step 3: Implement the box plot**

Append to `benchmark/plot.py`:

```python
def plot_latency_boxplot(rows, out_path):
    models = sorted({r["model"] for r in rows})
    devices = [d for d in ("cpu", "cuda") if any(r["device"] == d for r in rows)]
    fig, ax = plt.subplots()
    data, positions, labels = [], [], []
    width = 0.8 / max(len(devices), 1)
    for i, model in enumerate(models):
        for j, dev in enumerate(devices):
            samples = [r["latency_ms"] for r in rows
                       if r["model"] == model and r["device"] == dev]
            if samples:
                data.append(samples)
                positions.append(i + j * width)
                labels.append(f"{model}\n{dev}")
    if data:
        ax.boxplot(data, positions=positions, widths=width * 0.9)
        # Log y: CPU and GPU latencies span orders of magnitude.
        ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize="x-small")
    ax.set_ylabel("latency (ms), per image @640")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plot.py -v`
Expected: PASS (new test + the three pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add benchmark/plot.py tests/test_plot.py
git commit -m "feat(bench): add latency box-plot generator (Phase 2c)"
```

---

### Task 7: wire the new assets into `build_report_assets`

**Files:**
- Modify: `benchmark/build_report_assets.py`
- Test: `tests/test_build_report_assets.py`

**Interfaces:**
- Consumes: `latex_tables.feasibility_table` + `distribution_table` (Task 5), `plot.plot_latency_boxplot` (Task 6), `combine.load_dist` (Task 3), existing `load_combined`.
- Produces: `build(cpu_csv, cuda_csv, fig_dir, tex_dir, dist_cpu=None, dist_cuda=None)` — additionally writes `feasibility_table.tex` always, and `distribution_table.tex` + `latency_boxplot.pdf` when a distribution CSV is present. CLI gains `--dist-cpu`/`--dist-cuda`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_report_assets.py` (the file already imports `csv` and defines `_COLS`/`_write`):

```python
_DIST_COLS = ["device", "model", "task", "image_id", "latency_ms", "resolution", "measured_at"]


def _write_dist(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_DIST_COLS)
        w.writerows(rows)


def test_build_writes_feasibility_and_distribution_assets(tmp_path):
    cpu = tmp_path / "cpu.csv"
    mem = ["set", 640, "peak_rss", "rss_mb", 468.0, 1, "2026-07-01"]
    lat = ["set", 640, "warm_latency", "mean_ms", 200.0, 50, "2026-07-01"]
    _write(cpu, [["cpu", "yolo11n", "detection", *mem],
                 ["cpu", "yolo11n", "detection", *lat]])
    dist = tmp_path / "results_dist_cpu.csv"
    _write_dist(dist, [["cpu", "yolo11n", "detection", f"{i:012d}", 200.0 + i, 640, "2026-07-01"]
                       for i in range(10)])
    figd = tmp_path / "figures"
    texd = tmp_path / "generated"
    bra.build(cpu, None, figd, texd, dist_cpu=dist, dist_cuda=None)
    assert (texd / "feasibility_table.tex").exists()
    assert (texd / "distribution_table.tex").exists()
    assert (figd / "latency_boxplot.pdf").exists()
    assert "Yes" in (texd / "feasibility_table.tex").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_report_assets.py -v`
Expected: FAIL (`TypeError: build() got an unexpected keyword argument 'dist_cpu'`)

- [ ] **Step 3: Implement the wiring**

Edit `benchmark/build_report_assets.py`. Change the imports to add `load_dist`:

```python
from benchmark.combine import load_combined, load_dist
```

Replace the `build(...)` signature and body with:

```python
def build(cpu_csv, cuda_csv, fig_dir, tex_dir, dist_cpu=None, dist_cuda=None):
    fig_dir = pathlib.Path(fig_dir)
    tex_dir = pathlib.Path(tex_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    paths = [p for p in (cpu_csv, cuda_csv) if p and pathlib.Path(p).exists()]
    rows = load_combined(*paths)

    plot.plot_cpu_vs_gpu(rows, fig_dir / "cpu_vs_gpu.pdf")
    plot.plot_scaling(rows, fig_dir / "latency_scaling.pdf")
    plot.plot_memory_scaling(rows, fig_dir / "memory_scaling.pdf")

    (tex_dir / "perf_det_table.tex").write_text(lt.latency_table(rows, "detection", "tab:perf-det"))
    (tex_dir / "perf_seg_table.tex").write_text(lt.latency_table(rows, "segmentation", "tab:perf-seg"))
    (tex_dir / "coldwarm_det_table.tex").write_text(lt.coldwarm_table(rows, "detection", "tab:coldwarm-det"))
    (tex_dir / "coldwarm_seg_table.tex").write_text(lt.coldwarm_table(rows, "segmentation", "tab:coldwarm-seg"))
    (tex_dir / "memory_table.tex").write_text(lt.memory_table(rows, "tab:perf-mem"))
    (tex_dir / "feasibility_table.tex").write_text(lt.feasibility_table(rows, "tab:feasibility"))

    dist_paths = [p for p in (dist_cpu, dist_cuda) if p and pathlib.Path(p).exists()]
    if dist_paths:
        drows = load_dist(*dist_paths)
        (tex_dir / "distribution_table.tex").write_text(
            lt.distribution_table(drows, "tab:latency-dist"))
        plot.plot_latency_boxplot(drows, fig_dir / "latency_boxplot.pdf")
```

Replace `main(...)` with (adds the two dist args and passes them through):

```python
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build report figures + table fragments from CSVs")
    ap.add_argument("--cpu", default="data/phase2/results_cpu.csv")
    ap.add_argument("--cuda", default="data/phase2/results_cuda.csv")
    ap.add_argument("--dist-cpu", default="data/phase2/results_dist_cpu.csv")
    ap.add_argument("--dist-cuda", default="data/phase2/results_dist_cuda.csv")
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--tex-dir", required=True)
    args = ap.parse_args(argv)
    build(args.cpu, args.cuda, args.fig_dir, args.tex_dir,
          dist_cpu=args.dist_cpu, dist_cuda=args.dist_cuda)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_report_assets.py -v`
Expected: PASS (new test + the pre-existing one)

- [ ] **Step 5: Commit**

```bash
git add benchmark/build_report_assets.py tests/test_build_report_assets.py
git commit -m "feat(bench): emit feasibility + distribution assets in build_report_assets (Phase 2c)"
```

---

### Task 8: run the real CPU distribution campaign + Colab runbook note

**Files:**
- Create: `data/phase2/results_dist_cpu.csv` (generated artifact, committed)
- Create: `data/phase2/manifest_dist_cpu.json` (generated artifact, committed)
- Modify: `docs/COLAB_RUNBOOK.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7 (the full harness).
- Produces: the committed CPU distribution CSV + manifest, and a documented GPU handoff command.

- [ ] **Step 1: Run the full test suite (all green before a real campaign)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — the whole suite (previously 55, now higher with the Phase 2c tests), 0 failures.

- [ ] **Step 2: Run the CPU distribution campaign**

This downloads coco128 (~7 MB) on first use and loads real models (CPU torch). It is slow (minutes) but not interactive.

Run: `.venv/bin/python -m benchmark.run_dist --device cpu`
Expected: writes `data/phase2/results_dist_cpu.csv` and `data/phase2/manifest_dist_cpu.json`. Any `[skip]` lines name models that failed to load (e.g. missing optional weights) — acceptable, the rest still write.

- [ ] **Step 3: Sanity-check the output**

Run: `.venv/bin/python -c "from benchmark.combine import load_dist; r=load_dist('data/phase2/results_dist_cpu.csv'); import collections; c=collections.Counter(x['model'] for x in r); print(c); print('total', len(r))"`
Expected: each surviving model has ~100 rows (one per image); `total` ≈ 100 × (number of models that loaded).

- [ ] **Step 4: Regenerate assets locally to confirm they build (throwaway output dir)**

Run: `.venv/bin/python -m benchmark.build_report_assets --fig-dir /tmp/p2c_fig --tex-dir /tmp/p2c_tex`
Expected: `/tmp/p2c_tex/distribution_table.tex`, `/tmp/p2c_tex/feasibility_table.tex`, and `/tmp/p2c_fig/latency_boxplot.pdf` exist. (These go to `/tmp`; the report repo — Plan B — owns the real output paths.)

- [ ] **Step 5: Document the GPU handoff in the Colab runbook**

Add a short section to `docs/COLAB_RUNBOOK.md` telling the author to run the distribution campaign on the T4 after the existing `benchmark.run` step:

```markdown
## Phase 2c — latency distribution (GPU)

After the main campaign, run the distribution pass on the same T4 to get the GPU
percentiles and box-plot data:

```bash
python -m benchmark.run_dist --device cuda
```

This downloads coco128 (~7 MB) and writes `data/phase2/results_dist_cuda.csv` +
`manifest_dist_cuda.json`. Download both alongside the existing `results_cuda.csv`.
```

- [ ] **Step 6: Commit the CPU artifacts and the runbook note**

```bash
git add data/phase2/results_dist_cpu.csv data/phase2/manifest_dist_cpu.json docs/COLAB_RUNBOOK.md
git commit -m "data(bench): CPU latency-distribution campaign + GPU runbook note (Phase 2c)"
```

---

## Self-Review

**Spec coverage:**
- 100-image set → Task 1 (coco128 fetch). ✅ (deviation from val2017 documented in Global Constraints)
- Percentile primitive (mean/std/P50/P90/P99) → Task 2. ✅
- Distribution campaign writing raw per-image samples → Task 4 (+ loader Task 3). ✅
- CPU run → Task 8; GPU run → Task 8 Step 5 (author handoff). ✅
- 1 GB feasibility from isolated peak_rss (`< 1024`) → Task 5 (`feasibility_table`). ✅
- Distribution table + box-plot → Tasks 5 + 6, wired in Task 7. ✅
- Paper integration (Results subsection, threats, abstract/method) → **Plan B**, out of scope here. ✅
- Out-of-scope items (ONNX, own accuracy, PQ/mIoU, prompts) → not implemented, correct. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step gives an exact command + expected result. ✅

**Type consistency:** `latency_stats` keys (`n, mean_ms, std_ms, p50_ms, p90_ms, p99_ms`) are produced in Task 2 and consumed identically in Task 5. `_DIST_COLUMNS`/schema identical across Tasks 3, 4, 7 tests. `time_per_image(predict_one, images, *, warmup)` signature identical in Tasks 2 and 4. `load_dist` produces `latency_ms: float`, consumed by Tasks 5/6/7. ✅

**Scope:** One subsystem (the harness), one repo, 8 cohesive tasks ending in a runnable CPU campaign. ✅
