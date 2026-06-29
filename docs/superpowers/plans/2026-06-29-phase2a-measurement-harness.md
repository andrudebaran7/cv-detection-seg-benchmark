# Phase 2a: Measurement Harness + CPU Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, device-agnostic performance-measurement harness in the
`cv-detection-seg-benchmark` repo that measures all 6 models (latency, memory, throughput,
image-size scaling), produces CSVs + a hardware manifest on CPU locally, and plots figures —
ready for the author to re-run on a cloud GPU.

**Architecture:** A new `benchmark/` package, separate from the Streamlit `app/`, reusing the
`models/` wrappers via the `DetectionSegModel` interface. Wrappers gain an optional `device`
parameter (default preserves current CPU/auto behavior). The harness orchestrates
(model × image × resolution × experiment), writes long-format CSVs under `data/phase2/`, and a
plotting module renders figures from those CSVs into the report repo's `figures/`.

**Tech Stack:** Python 3.9+, pytest (mocked-wrapper pattern already in `tests/`), `psutil`
(CPU RSS), `torch.cuda` (GPU mem), `numpy`, `Pillow`, `matplotlib` (plots).

## Global Constraints

- Work only in repo `cv-detection-seg-benchmark` on a dedicated branch
  (`phase2-performance-campaign` already exists).
- Performance axis ONLY — no accuracy/mAP computation. Accuracy stays published.
- All 6 models: YOLO11n (det), YOLO11n-seg (seg), SAM 2 tiny (seg, prompted), RF-DETR-nano
  (det), YOLO-World (det, open-vocab), Mask2Former (seg, panoptic).
- The `device` parameter default MUST be `None` and preserve current behavior so the Streamlit
  app and existing tests are unaffected.
- Tests follow the repo's existing pattern: mock the underlying library by injecting a fake
  module into `sys.modules` (see `tests/test_yolo_wrapper.py`); no weight downloads in tests.
- Report mean ± std over a fixed iteration count with explicit warmup; never a single value.
- The harness must run identically with `--device cpu` and `--device cuda` (same code path).
- Every CSV is accompanied by a hardware manifest JSON.
- Keep the existing 22-test suite green at every commit (`pytest -q` from repo root).
- Do NOT add GPU/CUDA wheels to `requirements.txt` (Streamlit Cloud is CPU-only); harness-only
  deps go in `requirements-dev.txt`.

## File structure

| File | Responsibility |
|------|----------------|
| `data/benchmark_images/` | The fixed ~8-image set + `SOURCES.md` provenance |
| `benchmark/__init__.py` | Package marker |
| `benchmark/images.py` | Load the image set; resize to a target resolution |
| `benchmark/measure.py` | Warmup + timed iterations (mean/median/std); peak-memory capture |
| `benchmark/manifest.py` | Capture host hardware/software → dict/JSON |
| `benchmark/models_registry.py` | Map model key → (wrapper factory, task, predict-kwargs) |
| `benchmark/run.py` | Orchestrate experiments B1–B5; write CSVs; `__main__` CLI |
| `benchmark/plot.py` | Render figures from CSVs into the report `figures/` dir |
| `models/*_wrapper.py` | Add optional `device` param (6 files) |
| `tests/test_*` | One test module per new `benchmark/` module + device tests |
| `requirements-dev.txt` | Add `psutil`, `matplotlib` |
| `docs/COLAB_RUNBOOK.md` | How the author runs the GPU pass on Colab |

---

### Task 1: Benchmark image set

**Files:**
- Create: `data/benchmark_images/SOURCES.md`
- Create: `benchmark/__init__.py` (empty package marker)
- Create: `benchmark/fetch_images.py`
- Test: `tests/test_fetch_images.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `data/benchmark_images/` populated with `.jpg` files; `fetch_images.IMAGE_IDS`
  (list[int]); `fetch_images.target_dir() -> pathlib.Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_images.py
from benchmark import fetch_images


def test_image_ids_are_at_least_eight_and_unique():
    ids = fetch_images.IMAGE_IDS
    assert len(ids) >= 8
    assert len(set(ids)) == len(ids)


def test_url_for_builds_coco_val2017_url():
    assert fetch_images.url_for(139) == "http://images.cocodataset.org/val2017/000000000139.jpg"


def test_target_dir_points_at_repo_data():
    p = fetch_images.target_dir()
    assert p.name == "benchmark_images"
    assert p.parent.name == "data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fetch_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/__init__.py
```

```python
# benchmark/fetch_images.py
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
```

```markdown
<!-- data/benchmark_images/SOURCES.md -->
# Benchmark image set

Eight images from the COCO val2017 split, fetched by `benchmark/fetch_images.py`
from the official URL pattern `http://images.cocodataset.org/val2017/<id>.jpg`.
COCO images are licensed per the COCO terms (annotations CC BY 4.0; images per their
Flickr sources). Used here only for measuring inference latency/memory — no labels are
used and no accuracy is computed.

IDs: 139, 285, 632, 724, 776, 785, 802, 872
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fetch_images.py -v`
Expected: PASS (3 passed). These tests do NOT hit the network (they check constants/URLs/paths).

- [ ] **Step 5: Actually fetch the images (one-time, networked)**

Run: `.venv/bin/python -m benchmark.fetch_images`
Expected: prints 8 paths under `data/benchmark_images/`. Verify with
`ls data/benchmark_images/*.jpg | wc -l` → `8`.
If any download fails (network), report DONE_WITH_CONCERNS naming the failed IDs — the author
can supply images manually; do not block.

- [ ] **Step 6: Commit**

```bash
git add benchmark/__init__.py benchmark/fetch_images.py tests/test_fetch_images.py data/benchmark_images/
git commit -m "feat(benchmark): add fixed COCO val2017 image set + fetcher"
```

---

### Task 2: Device support in the six wrappers

**Files:**
- Modify: `models/yolo_wrapper.py`, `models/yoloworld_wrapper.py`, `models/sam2_wrapper.py`,
  `models/rfdetr_wrapper.py`, `models/mask2former_wrapper.py` (and YOLO11n-seg uses
  `YoloWrapper` with seg weights — same file)
- Test: `tests/test_device_support.py`

**Interfaces:**
- Consumes: nothing.
- Produces: every wrapper's `__init__` accepts `device: str | None = None`, stored as
  `self.device`. ultralytics wrappers pass `device=self.device` into the prediction call;
  transformers wrapper calls `model.to(self.device)` when `device` is set; rfdetr passes
  `device=self.device` to its constructor when set. `device=None` preserves current behavior.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_device_support.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np


def _fake_ultralytics_yolo():
    mod = types.ModuleType("ultralytics")
    result = MagicMock()
    result.names = {0: "person"}
    result.boxes.xyxy.tolist.return_value = [[1.0, 2.0, 3.0, 4.0]]
    result.boxes.conf.tolist.return_value = [0.9]
    result.boxes.cls.tolist.return_value = [0]
    result.masks = None
    instance = MagicMock(return_value=[result])
    mod.YOLO = MagicMock(return_value=instance)
    sys.modules["ultralytics"] = mod
    return instance


def test_yolo_threads_device_into_call():
    instance = _fake_ultralytics_yolo()
    from models.yolo_wrapper import YoloWrapper

    YoloWrapper(device="cuda").predict(np.zeros((8, 8, 3), dtype=np.uint8))
    # the model was called with device="cuda"
    _, kwargs = instance.call_args
    assert kwargs.get("device") == "cuda"


def test_yolo_default_device_is_none_call_omits_or_none():
    instance = _fake_ultralytics_yolo()
    from models.yolo_wrapper import YoloWrapper

    YoloWrapper().predict(np.zeros((8, 8, 3), dtype=np.uint8))
    _, kwargs = instance.call_args
    assert kwargs.get("device") is None


def test_mask2former_moves_model_to_device():
    mod = types.ModuleType("transformers")
    processor = MagicMock()
    processor.return_value = MagicMock()  # inputs object
    model = MagicMock()
    mod.AutoImageProcessor = MagicMock()
    mod.AutoImageProcessor.from_pretrained = MagicMock(return_value=processor)
    mod.Mask2FormerForUniversalSegmentation = MagicMock()
    mod.Mask2FormerForUniversalSegmentation.from_pretrained = MagicMock(return_value=model)
    sys.modules["transformers"] = mod

    from models.mask2former_wrapper import Mask2FormerWrapper

    w = Mask2FormerWrapper(device="cuda")
    w._load()
    model.to.assert_called_with("cuda")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_device_support.py -v`
Expected: FAIL — `YoloWrapper.__init__() got an unexpected keyword argument 'device'`.

- [ ] **Step 3: Implement device in the ultralytics wrappers**

In `models/yolo_wrapper.py`, change `__init__` and the call:

```python
    def __init__(self, weights: str = "yolo11n.pt", device: str | None = None) -> None:
        self.weights = weights
        self.device = device
        self._model = None
```

and in `predict`, pass the device into the call:

```python
        results = model(image, conf=conf, device=self.device, verbose=False)
```

In `models/yoloworld_wrapper.py`, same `__init__` addition (`device: str | None = None`,
`self.device = device`) and change its call to:

```python
        results = model(image, conf=conf, device=self.device, verbose=False)
```

In `models/sam2_wrapper.py`, same `__init__` addition and change its call to:

```python
        results = model(image, points=points, bboxes=bboxes, device=self.device, verbose=False)
```

- [ ] **Step 4: Implement device in the rfdetr wrapper**

In `models/rfdetr_wrapper.py`:

```python
    def __init__(self, model_name: str = "nano", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import rfdetr  # lazy import

            kwargs = {"device": self.device} if self.device is not None else {}
            self._model = rfdetr.RFDETRNano(**kwargs)
        return self._model
```

- [ ] **Step 5: Implement device in the mask2former wrapper**

In `models/mask2former_wrapper.py`:

```python
    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-tiny-coco-panoptic",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import (  # lazy import
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )

            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
            if self.device is not None:
                model.to(self.device)
            self._model = (processor, model)
        return self._model
```

And in `predict`, move inputs to the model's device after the processor call:

```python
        inputs = processor(images=image, return_tensors="pt")
        if self.device is not None:
            inputs = inputs.to(self.device)
        outputs = model(**inputs)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_device_support.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `.venv/bin/python -m pytest -q`
Expected: all green (existing 22 + new device tests). The default-`None` path keeps existing
wrapper tests passing.

- [ ] **Step 8: Commit**

```bash
git add models/ tests/test_device_support.py
git commit -m "feat(models): optional device param on all six wrappers (default unchanged)"
```

> Note: unit tests mock the libraries, so they verify the device is *threaded through*, not
> real GPU execution. Real CUDA correctness is validated by the author's Colab run (Phase 2a
> final step). If a library's real device API differs from the call above, fix it during the
> Colab run and report back.

---

### Task 3: Measurement primitives

**Files:**
- Create: `benchmark/measure.py`
- Test: `tests/test_measure.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `timeit_callable(fn, *, warmup=5, iters=50) -> dict` returning
    `{"n_iters": int, "mean_ms": float, "median_ms": float, "std_ms": float}`.
  - `peak_rss_mb() -> float` (current process resident set size in MB).
  - `time_first_call_ms(fn) -> float` (single call, for cold-start).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_measure.py
from benchmark import measure


def test_timeit_callable_runs_iters_and_reports_stats():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    stats = measure.timeit_callable(fn, warmup=2, iters=10)
    assert calls["n"] == 12  # warmup + iters
    assert stats["n_iters"] == 10
    assert stats["mean_ms"] >= 0.0
    assert stats["std_ms"] >= 0.0
    assert "median_ms" in stats


def test_peak_rss_mb_is_positive():
    assert measure.peak_rss_mb() > 0.0


def test_time_first_call_runs_once():
    calls = {"n": 0}
    measure.time_first_call_ms(lambda: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_measure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.measure'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/measure.py
from __future__ import annotations

import statistics
import time
from typing import Callable

import psutil


def timeit_callable(fn: Callable[[], object], *, warmup: int = 5, iters: int = 50) -> dict:
    for _ in range(warmup):
        fn()
    samples_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "n_iters": iters,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "std_ms": statistics.pstdev(samples_ms) if iters > 1 else 0.0,
    }


def time_first_call_ms(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_measure.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmark/measure.py tests/test_measure.py
git commit -m "feat(benchmark): timing + memory measurement primitives"
```

---

### Task 4: Hardware manifest

**Files:**
- Create: `benchmark/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_manifest(device: str) -> dict` with keys
  `device, cpu, ram_gb, gpu, torch_version, cuda_available, python, os, measured_at`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
from benchmark import manifest


def test_manifest_has_required_keys_and_device():
    m = manifest.build_manifest("cpu")
    for key in ["device", "cpu", "ram_gb", "gpu", "torch_version",
                "cuda_available", "python", "os", "measured_at"]:
        assert key in m
    assert m["device"] == "cpu"
    assert m["ram_gb"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.manifest'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/manifest.py
from __future__ import annotations

import datetime
import platform

import psutil


def _torch_info() -> tuple[str, bool, str]:
    try:
        import torch
    except Exception:
        return ("not-installed", False, "")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return (torch.__version__, torch.cuda.is_available(), gpu)


def build_manifest(device: str) -> dict:
    torch_version, cuda_available, gpu = _torch_info()
    return {
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "python": platform.python_version(),
        "os": platform.platform(),
        "measured_at": datetime.date.today().isoformat(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmark/manifest.py tests/test_manifest.py
git commit -m "feat(benchmark): hardware/software manifest capture"
```

---

### Task 5: Image loading + resolution sweep

**Files:**
- Create: `benchmark/images.py`
- Test: `tests/test_images.py`

**Interfaces:**
- Consumes: `data/benchmark_images/` (Task 1).
- Produces:
  - `RESOLUTIONS = [320, 640, 960, 1280]`
  - `load_images() -> list[PIL.Image.Image]` (all bundled images, RGB).
  - `resize(img, size: int) -> PIL.Image.Image` (square resize to `size`×`size`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_images.py
from PIL import Image

from benchmark import images


def test_resolutions_constant():
    assert images.RESOLUTIONS == [320, 640, 960, 1280]


def test_resize_returns_square_target():
    src = Image.new("RGB", (1000, 500))
    out = images.resize(src, 640)
    assert out.size == (640, 640)
    assert out.mode == "RGB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.images'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/images.py
from __future__ import annotations

import pathlib

from PIL import Image

RESOLUTIONS = [320, 640, 960, 1280]


def _dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "benchmark_images"


def load_images() -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in sorted(_dir().glob("*.jpg"))]


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_images.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmark/images.py tests/test_images.py
git commit -m "feat(benchmark): image loading and resolution-sweep helper"
```

---

### Task 6: Model registry

**Files:**
- Create: `benchmark/models_registry.py`
- Test: `tests/test_models_registry.py`

**Interfaces:**
- Consumes: the `models/` wrappers (Task 2).
- Produces:
  - `ModelSpec` dataclass: `key: str, task: str, factory: Callable[[str | None], DetectionSegModel], predict_kwargs: Callable[[PIL.Image.Image], dict]`.
  - `REGISTRY: dict[str, ModelSpec]` with the six keys: `yolo11n`, `yolo11n-seg`,
    `sam2-tiny`, `rfdetr-nano`, `yolo-world`, `mask2former`.
  - `factory(device)` builds the wrapper on `device`; `predict_kwargs(img)` returns the
    per-model kwargs (e.g. SAM 2 center-point prompt, YOLO-World classes).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_registry.py
from PIL import Image

from benchmark import models_registry as reg


def test_registry_has_six_models_with_tasks():
    assert set(reg.REGISTRY) == {
        "yolo11n", "yolo11n-seg", "sam2-tiny", "rfdetr-nano", "yolo-world", "mask2former",
    }
    tasks = {s.task for s in reg.REGISTRY.values()}
    assert tasks == {"detection", "segmentation"}


def test_sam2_predict_kwargs_supply_center_point():
    spec = reg.REGISTRY["sam2-tiny"]
    kwargs = spec.predict_kwargs(Image.new("RGB", (640, 480)))
    assert "points" in kwargs and kwargs["points"] == [[320, 240]]


def test_yoloworld_predict_kwargs_supply_classes():
    spec = reg.REGISTRY["yolo-world"]
    kwargs = spec.predict_kwargs(Image.new("RGB", (640, 480)))
    assert kwargs.get("classes")  # non-empty list of text classes


def test_factory_passes_device_through():
    import sys, types
    from unittest.mock import MagicMock
    mod = types.ModuleType("ultralytics")
    mod.YOLO = MagicMock(return_value=MagicMock())
    sys.modules["ultralytics"] = mod
    model = reg.REGISTRY["yolo11n"].factory("cuda")
    assert model.device == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_models_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.models_registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/models_registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from models.base import DetectionSegModel

_COCO_CLASSES = ["person", "car", "dog", "chair", "bottle"]  # text prompts for YOLO-World


@dataclass
class ModelSpec:
    key: str
    task: str
    factory: Callable[[str | None], DetectionSegModel]
    predict_kwargs: Callable[[Image.Image], dict]


def _center_point(img: Image.Image) -> dict:
    w, h = img.size
    return {"points": [[w // 2, h // 2]]}


def _yolo11n(device):
    from models.yolo_wrapper import YoloWrapper
    return YoloWrapper(device=device)


def _yolo11n_seg(device):
    from models.yolo_wrapper import YoloWrapper
    return YoloWrapper(weights="yolo11n-seg.pt", device=device)


def _sam2(device):
    from models.sam2_wrapper import Sam2Wrapper
    return Sam2Wrapper(device=device)


def _rfdetr(device):
    from models.rfdetr_wrapper import RfDetrWrapper
    return RfDetrWrapper(device=device)


def _yoloworld(device):
    from models.yoloworld_wrapper import YoloWorldWrapper
    return YoloWorldWrapper(device=device)


def _mask2former(device):
    from models.mask2former_wrapper import Mask2FormerWrapper
    return Mask2FormerWrapper(device=device)


REGISTRY: dict[str, ModelSpec] = {
    "yolo11n": ModelSpec("yolo11n", "detection", _yolo11n, lambda img: {}),
    "yolo11n-seg": ModelSpec("yolo11n-seg", "segmentation", _yolo11n_seg, lambda img: {}),
    "sam2-tiny": ModelSpec("sam2-tiny", "segmentation", _sam2, _center_point),
    "rfdetr-nano": ModelSpec("rfdetr-nano", "detection", _rfdetr, lambda img: {}),
    "yolo-world": ModelSpec("yolo-world", "detection", _yoloworld,
                            lambda img: {"classes": _COCO_CLASSES}),
    "mask2former": ModelSpec("mask2former", "segmentation", _mask2former, lambda img: {}),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models_registry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmark/models_registry.py tests/test_models_registry.py
git commit -m "feat(benchmark): model registry (six models, tasks, per-model predict kwargs)"
```

---

### Task 7: Campaign orchestration + CLI

**Files:**
- Create: `benchmark/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `measure`, `manifest`, `images`, `models_registry`.
- Produces:
  - `run_model(spec, image, *, device, resolution, iters, warmup) -> list[dict]` returning
    one row dict per metric (experiments B1/B3/B4/B5 produce warm latency + throughput + rss;
    B2 produces a cold-start row from a fresh model).
  - `write_csv(rows: list[dict], path)` (long format, fixed column order).
  - `main(argv=None)` CLI: `--device {cpu,cuda}`, `--models all|<key,...>`, `--iters`,
    `--warmup`, `--out <dir>`; writes `<out>/results_<device>.csv` and
    `<out>/manifest_<device>.json`.
- Row schema (column order):
  `device, model, task, image, resolution, experiment, metric, value, n_iters, measured_at`.

- [ ] **Step 1: Write the failing test (orchestration with a fake model)**

```python
# tests/test_run.py
import csv

from PIL import Image

from benchmark import run
from benchmark.models_registry import ModelSpec


class _FakeModel:
    def __init__(self, device=None):
        self.device = device
    def predict(self, image, **kwargs):
        class P:  # minimal Prediction-like
            latency_ms = 1.0
        return P()


def _fake_spec():
    return ModelSpec("fake", "detection", lambda device: _FakeModel(device), lambda img: {})


def test_run_model_emits_rows_for_each_experiment():
    img = Image.new("RGB", (640, 480))
    rows = run.run_model(_fake_spec(), img, device="cpu", resolution=640, iters=3, warmup=1)
    experiments = {r["experiment"] for r in rows}
    assert {"warm_latency", "cold_start", "throughput", "peak_rss"} <= experiments
    for r in rows:
        assert r["device"] == "cpu"
        assert r["model"] == "fake"
        assert r["resolution"] == 640
        assert isinstance(r["value"], float)


def test_write_csv_has_fixed_header(tmp_path):
    rows = [{"device": "cpu", "model": "fake", "task": "detection", "image": "x",
             "resolution": 640, "experiment": "warm_latency", "metric": "mean_ms",
             "value": 1.0, "n_iters": 3, "measured_at": "2026-06-29"}]
    p = tmp_path / "r.csv"
    run.write_csv(rows, p)
    with open(p) as f:
        header = next(csv.reader(f))
    assert header == ["device", "model", "task", "image", "resolution",
                      "experiment", "metric", "value", "n_iters", "measured_at"]


def test_main_writes_csv_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REGISTRY", {"fake": _fake_spec()})
    monkeypatch.setattr(run, "load_images", lambda: [Image.new("RGB", (64, 64))])
    monkeypatch.setattr(run, "RESOLUTIONS", [320])
    run.main(["--device", "cpu", "--models", "all", "--iters", "2",
              "--warmup", "1", "--out", str(tmp_path)])
    assert (tmp_path / "results_cpu.csv").exists()
    assert (tmp_path / "manifest_cpu.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/run.py
from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib

from benchmark.images import RESOLUTIONS, load_images, resize
from benchmark.manifest import build_manifest
from benchmark.measure import peak_rss_mb, time_first_call_ms, timeit_callable
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

    # Warm model reused for B1/B3/B5.
    model = spec.factory(device)
    kwargs = spec.predict_kwargs(img)
    call = lambda: model.predict(img, **kwargs)

    stats = timeit_callable(call, warmup=warmup, iters=iters)
    rows.append(_row(spec, device, resolution, "warm_latency", "mean_ms", stats["mean_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "median_ms", stats["median_ms"], iters))
    rows.append(_row(spec, device, resolution, "warm_latency", "std_ms", stats["std_ms"], iters))

    throughput = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else 0.0
    rows.append(_row(spec, device, resolution, "throughput", "imgs_per_sec", throughput, iters))

    rows.append(_row(spec, device, resolution, "peak_rss", "rss_mb", peak_rss_mb(), iters))
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
    for key in keys:
        spec = REGISTRY[key]
        for resolution in RESOLUTIONS:
            rows.extend(run_model(spec, base, device=args.device,
                                  resolution=resolution, iters=args.iters, warmup=args.warmup))

    write_csv(rows, out / f"results_{args.device}.csv")
    with open(out / f"manifest_{args.device}.json", "w") as f:
        json.dump(build_manifest(args.device), f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_run.py -v`
Expected: PASS (3 passed). The `main` test monkeypatches `REGISTRY`, `load_images`,
`RESOLUTIONS` so it never loads a real model or hits the network.

- [ ] **Step 5: Commit**

```bash
git add benchmark/run.py tests/test_run.py
git commit -m "feat(benchmark): campaign orchestration + CLI (B1-B5 -> CSV + manifest)"
```

---

### Task 8: Plotting

**Files:**
- Create: `benchmark/plot.py`
- Test: `tests/test_plot.py`

**Interfaces:**
- Consumes: a results CSV (Task 7 schema).
- Produces:
  - `load_rows(path) -> list[dict]`.
  - `plot_cpu_vs_gpu(rows, out_path)` (bar chart, warm_latency mean_ms at resolution 640).
  - `plot_scaling(rows, out_path)` (line chart, warm_latency mean_ms vs resolution per model).
  - Each writes a file at `out_path`; `out_path` may be under the report `figures/` dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot.py
import csv

from benchmark import plot


def _write_rows(path):
    cols = ["device", "model", "task", "image", "resolution",
            "experiment", "metric", "value", "n_iters", "measured_at"]
    rows = [
        ["cpu", "yolo11n", "detection", "set", 640, "warm_latency", "mean_ms", 200.0, 50, "2026-06-29"],
        ["cpu", "yolo11n", "detection", "set", 320, "warm_latency", "mean_ms", 90.0, 50, "2026-06-29"],
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def test_load_rows_parses_numeric_fields(tmp_path):
    p = tmp_path / "r.csv"
    _write_rows(p)
    rows = plot.load_rows(p)
    assert rows[0]["value"] == 200.0
    assert rows[0]["resolution"] == 640


def test_plot_scaling_writes_file(tmp_path):
    p = tmp_path / "r.csv"
    _write_rows(p)
    out = tmp_path / "scaling.png"
    plot.plot_scaling(plot.load_rows(p), out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.plot'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/plot.py
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def load_rows(path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["resolution"] = int(r["resolution"])
            r["value"] = float(r["value"])
            rows.append(r)
    return rows


def _filter(rows, **eq):
    return [r for r in rows if all(r[k] == v for k, v in eq.items())]


def plot_cpu_vs_gpu(rows, out_path):
    sel = _filter(rows, experiment="warm_latency", metric="mean_ms", resolution=640)
    models = sorted({r["model"] for r in sel})
    fig, ax = plt.subplots()
    for i, device in enumerate(["cpu", "cuda"]):
        vals = [next((r["value"] for r in sel if r["model"] == m and r["device"] == device), 0.0)
                for m in models]
        ax.bar([x + i * 0.4 for x in range(len(models))], vals, width=0.4, label=device)
    ax.set_xticks([x + 0.2 for x in range(len(models))])
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("warm latency (ms), res 640")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scaling(rows, out_path):
    sel = _filter(rows, experiment="warm_latency", metric="mean_ms")
    fig, ax = plt.subplots()
    for model in sorted({r["model"] for r in sel}):
        pts = sorted((r["resolution"], r["value"]) for r in sel if r["model"] == model)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model)
    ax.set_xlabel("resolution (px)")
    ax.set_ylabel("warm latency (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_plot.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmark/plot.py tests/test_plot.py
git commit -m "feat(benchmark): plotting (cpu-vs-gpu bars, latency-vs-resolution lines)"
```

---

### Task 9: Dev deps, Colab runbook, real CPU run, full suite

**Files:**
- Modify: `requirements-dev.txt`
- Create: `docs/COLAB_RUNBOOK.md`
- Create (output, not committed by tests): `data/phase2/results_cpu.csv`, `data/phase2/manifest_cpu.json`

**Interfaces:**
- Consumes: everything above.
- Produces: a real CPU results CSV + manifest, and the runbook the author follows on Colab.

- [ ] **Step 1: Add harness dev deps**

Append to `requirements-dev.txt`:

```
psutil>=5.9
matplotlib>=3.7
```

- [ ] **Step 2: Install and run the FULL suite**

Run: `.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest -q`
Expected: all green (the original 22 + the new harness/device tests). Record the new total.

- [ ] **Step 3: Run the real CPU campaign (small iters for a sanity pass)**

Run: `.venv/bin/python -m benchmark.run --device cpu --models all --iters 10 --warmup 3 --out data/phase2`
Expected: writes `data/phase2/results_cpu.csv` and `data/phase2/manifest_cpu.json`. First run
downloads weights (slow). If a model errors on real CPU (e.g. a real device API mismatch),
fix the wrapper and re-run; report the fix. Then optionally re-run with `--iters 50` for the
real numbers.

- [ ] **Step 4: Smoke-render a figure from the real CSV**

Run:
```bash
.venv/bin/python -c "from benchmark import plot; rows=plot.load_rows('data/phase2/results_cpu.csv'); plot.plot_scaling(rows, 'data/phase2/scaling_cpu.png'); print('ok')"
```
Expected: prints `ok`; `data/phase2/scaling_cpu.png` exists.

- [ ] **Step 5: Write the Colab runbook**

```markdown
<!-- docs/COLAB_RUNBOOK.md -->
# Phase 2 — running the GPU pass on Colab

1. Open a GPU runtime (Runtime → Change runtime type → GPU).
2. Clone the repo and install deps (GPU torch is preinstalled on Colab):
   `!git clone <repo-url> && cd cv-detection-seg-benchmark && pip install -r requirements-dev.txt`
3. Fetch the image set: `!python -m benchmark.fetch_images`
4. Run the campaign on GPU:
   `!python -m benchmark.run --device cuda --models all --iters 50 --warmup 5 --out data/phase2`
5. Download `data/phase2/results_cuda.csv` and `data/phase2/manifest_cuda.json`, then commit
   them next to the CPU CSVs. Plan 2b merges both into the paper's tables and figures.
```

- [ ] **Step 6: Commit (code + runbook + CPU data)**

```bash
git add requirements-dev.txt docs/COLAB_RUNBOOK.md data/phase2/results_cpu.csv data/phase2/manifest_cpu.json
git commit -m "chore(benchmark): dev deps, Colab runbook, first CPU campaign results"
```

> Human handoff after this task: the author runs `docs/COLAB_RUNBOOK.md` on Colab to produce
> `results_cuda.csv` + `manifest_cuda.json` and commits them. Only then does Plan 2b (paper
> integration) begin.

---

## Self-Review

**Spec coverage:**
- Portable harness (`benchmark/` separate from app) → Tasks 3–8.
- Device support in wrappers (default unchanged) → Task 2.
- Fixed ~8-image set + resolution sweep → Tasks 1, 5.
- All 6 models, det/seg tasks → Task 6 registry.
- Experiments B1 (warm latency), B2 (cold-start), B3 (throughput), B4 (scaling), B5 (memory)
  → Task 7 `run_model` emits rows for each; B4 via the resolution loop in `main`.
- CSV long-format schema + hardware manifest → Tasks 4, 7.
- Mean ± std over fixed iters with warmup → Task 3 `timeit_callable`.
- Data-plot figures → Task 8.
- CPU data produced; GPU pass handed off via runbook → Task 9.
- Dev-only deps (psutil/matplotlib), not in requirements.txt → Task 9 Step 1.
- Keep suite green → Task 2 Step 7, Task 9 Step 2.

**Out of scope (correctly absent):** accuracy/mAP, ONNX/TensorRT, multi-GPU, extra datasets,
extra models, video — all deferred to Future Work per the spec. Architecture/flow diagrams and
app screenshots are figures handled in Plan 2b (paper integration), not here.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the only "report and
continue" branches (Task 1 Step 5 download failure, Task 9 Step 3 real-device mismatch) name
the concrete action.

**Type consistency:** the CSV `_COLUMNS` order in Task 7 matches the header asserted in
`tests/test_run.py` and `tests/test_plot.py`; `ModelSpec` fields (`key, task, factory,
predict_kwargs`) are consistent across Tasks 6, 7, 8; `timeit_callable` keys
(`n_iters, mean_ms, median_ms, std_ms`) match their consumers in `run.run_model`.
