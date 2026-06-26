# cv-detection-seg-benchmark v0.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Mask2Former (panoptic segmentation) wrapper and a benchmark page (measured latency vs published mAP) to the existing benchmark, through the unified `DetectionSegModel` interface.

**Architecture:** `Mask2FormerWrapper` implements the existing `DetectionSegModel` ABC, using HuggingFace `transformers` to run `facebook/mask2former-resnet50-coco-panoptic` and mapping each panoptic segment to the existing `Prediction` (one mask per segment, `boxes=[]`). A pure `benchmark` helper measures each detector's `latency_ms` and joins it with a static `PUBLISHED_MAP` table; a new Streamlit page renders an `st.scatter_chart` plus a table. Heavy backends are lazy-imported; all wrapper/helper tests mock the backend.

**Tech Stack:** Python 3.13, transformers, torch, ultralytics, rfdetr, streamlit, numpy, pillow, pytest.

## Global Constraints

- Python >= 3.9. Use the existing project venv `.venv`.
- Repo license remains **AGPL-3.0**; new dep `transformers` is Apache-2.0.
- Mask2Former uses **`facebook/mask2former-resnet50-coco-panoptic`** (~170 MB), not Swin-L.
- All wrappers MUST **lazy-import** their backend (import inside methods, never at module top).
- Tests MUST **mock** the backend — never download weights in tests.
- `Prediction` (from `models/base.py`) remains the only data contract.
- Benchmark page is **detection-focused**; Mask2Former is not placed on the mAP scatter.
- No new plotting dependency: use Streamlit's native `st.scatter_chart`.
- Commit message bodies end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Run git commits with `git -c commit.gpgsign=false`.

---

## File Structure

- `models/mask2former_wrapper.py` (new) — `Mask2FormerWrapper(DetectionSegModel)`, panoptic → Prediction.
- `app/components/benchmark.py` (new) — `PUBLISHED_MAP`, `measure_latency`, `build_benchmark_rows`.
- `app/components/model_runner.py` (modify) — add `get_mask2former`.
- `app/pages/2_Segmentation.py` (modify) — add Mask2Former option.
- `app/pages/5_Benchmark.py` (new) — benchmark page.
- `requirements.txt`, `requirements-light.txt` (modify) — add `transformers`.
- `README.md` (modify) — model table + pages + roadmap.
- Tests: `tests/test_mask2former_wrapper.py`, `tests/test_benchmark.py` (new).

Work from: `/home/andru/Projects_claude/cv-detection-seg-benchmark`. Branch `main`.

---

## Task 1: Mask2Former wrapper (panoptic segmentation)

**Files:**
- Create: `models/mask2former_wrapper.py`
- Test: `tests/test_mask2former_wrapper.py`

**Interfaces:**
- Consumes: `DetectionSegModel`, `Prediction` from `models.base`.
- Produces:
  - `class Mask2FormerWrapper(DetectionSegModel)`, `__init__(self, model_id: str = "facebook/mask2former-resnet50-coco-panoptic")` storing `self.model_id`, `self._model = None`.
  - `_load(self)` — lazy-imports `transformers.AutoImageProcessor` and `transformers.Mask2FormerForUniversalSegmentation`, caches `self._model = (processor, model)`.
  - `predict(self, image, **kwargs) -> Prediction` — runs the model and `processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]`; for each entry in `segments_info` builds a boolean mask `(segmentation == seg["id"])`, label `model.config.id2label[seg["label_id"]]`, score `seg["score"]`; returns `Prediction(boxes=[], labels, scores, masks, latency_ms)`.

- [ ] **Step 1: Write the failing test (mock transformers)**

```python
# tests/test_mask2former_wrapper.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from models.base import Prediction


def _install_fake_transformers():
    mod = types.ModuleType("transformers")

    processor = MagicMock()
    # processor(images=..., return_tensors=...) -> dict-like inputs
    processor.return_value = {"pixel_values": MagicMock()}
    # post_process returns one dict: a 2x2 segmentation map with ids 1 and 2
    seg = np.array([[1, 1], [2, 2]])
    processor.post_process_panoptic_segmentation.return_value = [
        {
            "segmentation": seg,
            "segments_info": [
                {"id": 1, "label_id": 0, "score": 0.9},
                {"id": 2, "label_id": 5, "score": 0.8},
            ],
        }
    ]

    model = MagicMock()
    model.config.id2label = {0: "person", 5: "bus"}
    model.return_value = MagicMock()  # outputs

    mod.AutoImageProcessor = MagicMock()
    mod.AutoImageProcessor.from_pretrained = MagicMock(return_value=processor)
    mod.Mask2FormerForUniversalSegmentation = MagicMock()
    mod.Mask2FormerForUniversalSegmentation.from_pretrained = MagicMock(return_value=model)
    sys.modules["transformers"] = mod
    return processor, model


def test_mask2former_maps_panoptic_to_prediction():
    _install_fake_transformers()
    from models.mask2former_wrapper import Mask2FormerWrapper

    img = Image.new("RGB", (2, 2), "white")
    pred = Mask2FormerWrapper().predict(img)

    assert isinstance(pred, Prediction)
    assert pred.boxes == []
    assert pred.labels == ["person", "bus"]
    assert pred.scores == [0.9, 0.8]
    assert pred.masks is not None and len(pred.masks) == 2
    # first mask selects the top row (id == 1)
    assert np.array(pred.masks[0]).tolist() == [[True, True], [False, False]]
    assert pred.latency_ms >= 0.0


def test_mask2former_lazy_import():
    sys.modules.pop("transformers", None)
    import importlib
    import models.mask2former_wrapper as mw
    importlib.reload(mw)  # importing must not require transformers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mask2former_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.mask2former_wrapper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/mask2former_wrapper.py
from __future__ import annotations

import time
from typing import Any

import numpy as np

from models.base import DetectionSegModel, Prediction


class Mask2FormerWrapper(DetectionSegModel):
    def __init__(self, model_id: str = "facebook/mask2former-resnet50-coco-panoptic") -> None:
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import (  # lazy import
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )

            processor = AutoImageProcessor.from_pretrained(self.model_id)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
            self._model = (processor, model)
        return self._model

    def predict(self, image: Any, **kwargs: Any) -> Prediction:
        processor, model = self._load()
        pil = image if hasattr(image, "size") else None
        width, height = (pil.size if pil is not None else (image.shape[1], image.shape[0]))

        start = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        seg = np.array(result["segmentation"])
        labels: list[str] = []
        scores: list[float] = []
        masks: list[Any] = []
        for info in result["segments_info"]:
            masks.append((seg == info["id"]))
            labels.append(model.config.id2label[info["label_id"]])
            scores.append(float(info["score"]))

        return Prediction(
            boxes=[],
            labels=labels,
            scores=scores,
            masks=masks,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_mask2former_wrapper.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/mask2former_wrapper.py tests/test_mask2former_wrapper.py
git -c commit.gpgsign=false commit -m "feat(models): add Mask2FormerWrapper (panoptic) with mocked tests"
```

---

## Task 2: Benchmark helper

**Files:**
- Create: `app/components/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `DetectionSegModel` from `models.base`.
- Produces:
  - `PUBLISHED_MAP: dict[str, dict]` — keys `"YOLO11n"`, `"RF-DETR-nano"`, `"YOLO-World"`, each `{"map": float, "note": str}`.
  - `def measure_latency(model: DetectionSegModel, image, **kwargs) -> float` — returns `model.predict(image, **kwargs).latency_ms`.
  - `def build_benchmark_rows(measured: dict[str, float]) -> list[dict]` — for each name in `measured` order that is also in `PUBLISHED_MAP`, returns `{"model": name, "map": <map>, "latency_ms": <measured>, "note": <note>}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
from unittest.mock import MagicMock

from models.base import Box, Prediction
from app.components.benchmark import (
    PUBLISHED_MAP,
    measure_latency,
    build_benchmark_rows,
)


def test_published_map_has_three_detectors():
    assert set(PUBLISHED_MAP) == {"YOLO11n", "RF-DETR-nano", "YOLO-World"}
    for entry in PUBLISHED_MAP.values():
        assert "map" in entry and "note" in entry


def test_measure_latency_returns_prediction_latency():
    model = MagicMock()
    model.predict.return_value = Prediction(
        boxes=[Box(0, 0, 1, 1)], labels=["x"], scores=[0.5], latency_ms=12.0
    )
    assert measure_latency(model, object(), conf=0.3) == 12.0
    model.predict.assert_called_once()


def test_build_benchmark_rows_joins_and_preserves_order():
    measured = {"RF-DETR-nano": 30.0, "YOLO11n": 10.0, "Unknown": 99.0}
    rows = build_benchmark_rows(measured)

    assert [r["model"] for r in rows] == ["RF-DETR-nano", "YOLO11n"]  # Unknown dropped
    assert rows[0]["latency_ms"] == 30.0
    assert rows[1]["latency_ms"] == 10.0
    assert rows[0]["map"] == PUBLISHED_MAP["RF-DETR-nano"]["map"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.components.benchmark'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/components/benchmark.py
from __future__ import annotations

from typing import Any

from models.base import DetectionSegModel

# Published headline metrics for the detectors in the benchmark. COCO mAP unless noted.
PUBLISHED_MAP: dict[str, dict] = {
    "YOLO11n": {"map": 39.5, "note": "COCO val mAP (nano)"},
    "RF-DETR-nano": {"map": 48.4, "note": "COCO val mAP (nano)"},
    "YOLO-World": {"map": 35.4, "note": "LVIS zero-shot AP (small)"},
}


def measure_latency(model: DetectionSegModel, image: Any, **kwargs: Any) -> float:
    return model.predict(image, **kwargs).latency_ms


def build_benchmark_rows(measured: "dict[str, float]") -> "list[dict]":
    rows: list[dict] = []
    for name, latency in measured.items():
        if name in PUBLISHED_MAP:
            rows.append(
                {
                    "model": name,
                    "map": PUBLISHED_MAP[name]["map"],
                    "latency_ms": latency,
                    "note": PUBLISHED_MAP[name]["note"],
                }
            )
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (v0.1 + v0.2 + the two new files), output pristine.

- [ ] **Step 6: Commit**

```bash
git add app/components/benchmark.py tests/test_benchmark.py
git -c commit.gpgsign=false commit -m "feat(app): add benchmark helper (PUBLISHED_MAP, latency, rows)"
```

---

## Task 3: Dependency + Mask2Former loader

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-light.txt`
- Modify: `app/components/model_runner.py`

**Interfaces:**
- Consumes: `Mask2FormerWrapper`.
- Produces: `get_mask2former() -> Mask2FormerWrapper` (`@st.cache_resource`).

Verified by an end-to-end smoke test (no new unit test).

- [ ] **Step 1: Add dep to `requirements.txt`** (append after `supervision>=0.20`)

```
transformers>=4.40
```

- [ ] **Step 2: Add dep to `requirements-light.txt`** (append after `supervision>=0.20`)

```
transformers>=4.40
```

- [ ] **Step 3: Install transformers**

Run: `.venv/bin/pip install "transformers>=4.40"`
Expected: installs successfully.

- [ ] **Step 4: Add loader to `app/components/model_runner.py`** (add import + function)

Add this import next to the other model imports:

```python
from models.mask2former_wrapper import Mask2FormerWrapper
```

Append this function:

```python
@st.cache_resource
def get_mask2former(model_id: str = "facebook/mask2former-resnet50-coco-panoptic") -> Mask2FormerWrapper:
    return Mask2FormerWrapper(model_id)
```

- [ ] **Step 5: End-to-end Mask2Former smoke test (real weights)**

```bash
.venv/bin/python - <<'PY'
from PIL import Image
from models.mask2former_wrapper import Mask2FormerWrapper

img = Image.open("data/sample_images/bus.jpg").convert("RGB")
pred = Mask2FormerWrapper().predict(img)
print("TYPE", type(pred).__name__)
print("NUM_SEGMENTS", len(pred.masks) if pred.masks else 0)
print("LABELS", sorted(set(pred.labels))[:8])
print("BOXES_EMPTY", pred.boxes == [])
print("LATENCY_MS", round(pred.latency_ms))
PY
```

Expected: prints several panoptic segments with labels (e.g. `bus`, `person`, `road`) and `BOXES_EMPTY True`. If `post_process_panoptic_segmentation` output keys differ from `segmentation`/`segments_info`/`id`/`label_id`/`score`, report `DONE_WITH_CONCERNS` with the actual keys so Task 1's wrapper + test are corrected.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-light.txt app/components/model_runner.py
git -c commit.gpgsign=false commit -m "feat(app): add transformers dep and Mask2Former loader"
```

---

## Task 4: Mask2Former on the Segmentation page

**Files:**
- Modify: `app/pages/2_Segmentation.py`

**Interfaces:**
- Consumes: `get_yolo_seg`, `get_sam2`, `get_mask2former`; `draw_prediction`.

Verified by launching the app (manual smoke test).

- [ ] **Step 1: Rewrite `app/pages/2_Segmentation.py`**

```python
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo_seg, get_sam2, get_mask2former
from app.components.visualization import draw_prediction

st.title("Segmentation — YOLO11n-seg / SAM 2 / Mask2Former")

mode = st.radio(
    "Model",
    ["YOLO11n-seg (automatic)", "SAM 2 tiny (point prompt)", "Mask2Former (panoptic)"],
)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)

    if mode.startswith("YOLO"):
        with st.spinner("Running YOLO11n-seg..."):
            pred = get_yolo_seg().predict(arr)
    elif mode.startswith("SAM"):
        st.caption("Click point is the image center in v0.1 (interactive click lands in v0.2).")
        h, w = arr.shape[:2]
        with st.spinner("Running SAM 2 tiny..."):
            pred = get_sam2().predict(arr, points=[[w // 2, h // 2]])
    else:
        with st.spinner("Running Mask2Former (panoptic)..."):
            pred = get_mask2former().predict(image)

    st.image(draw_prediction(image, pred), caption=f"{pred.latency_ms:.0f} ms")
```

- [ ] **Step 2: Smoke-test the app**

```bash
.venv/bin/streamlit run app/main.py --server.headless true --server.port 8770 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8770
```

Expected: `200`; then stop the server. Confirm the Segmentation page shows the three-option radio.

- [ ] **Step 3: Commit**

```bash
git add app/pages/2_Segmentation.py
git -c commit.gpgsign=false commit -m "feat(app): add Mask2Former option to segmentation page"
```

---

## Task 5: Benchmark page

**Files:**
- Create: `app/pages/5_Benchmark.py`

**Interfaces:**
- Consumes: `get_yolo`, `get_rfdetr`, `get_yoloworld`; `measure_latency`, `build_benchmark_rows`.

Verified by launching the app (manual smoke test).

- [ ] **Step 1: Write `app/pages/5_Benchmark.py`**

```python
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr, get_yoloworld
from app.components.benchmark import measure_latency, build_benchmark_rows

st.title("Benchmark — latency vs published mAP")
st.caption(
    "Latency is measured on this machine for the bundled sample image; mAP values are "
    "published headline numbers (YOLO-World is LVIS zero-shot)."
)

ALL = ["YOLO11n", "RF-DETR-nano", "YOLO-World"]
chosen = st.multiselect("Detectors", ALL, default=ALL)

if st.button("Run benchmark"):
    if len(chosen) < 2:
        st.warning("Pick at least two detectors.")
    else:
        image = Image.open("data/sample_images/bus.jpg").convert("RGB")
        arr = np.array(image)
        loaders = {"YOLO11n": get_yolo, "RF-DETR-nano": get_rfdetr, "YOLO-World": get_yoloworld}

        measured = {}
        with st.spinner("Measuring latency..."):
            for name in chosen:
                model = loaders[name]()
                if name == "YOLO-World":
                    measured[name] = measure_latency(model, arr, classes=["person", "car", "bus"])
                else:
                    measured[name] = measure_latency(model, arr)

        rows = build_benchmark_rows(measured)
        st.scatter_chart(
            {"latency_ms": [r["latency_ms"] for r in rows], "map": [r["map"] for r in rows]},
            x="latency_ms",
            y="map",
        )
        st.table(rows)
```

- [ ] **Step 2: Smoke-test**

```bash
.venv/bin/streamlit run app/main.py --server.headless true --server.port 8771 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8771
```

Expected: `200`; stop the server. Confirm the Benchmark page appears in the sidebar.

- [ ] **Step 3: Commit**

```bash
git add app/pages/5_Benchmark.py
git -c commit.gpgsign=false commit -m "feat(app): add benchmark page (latency vs mAP)"
```

---

## Task 6: README + roadmap update, full verify, push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README** — add a `Mask2Former (panoptic)` row to the model table (`facebook/mask2former-resnet50-coco-panoptic`, HuggingFace); add `Benchmark` to the Pages list; add `models/mask2former_wrapper.py` and `app/components/benchmark.py` to the architecture block; mark v0.3 done in the roadmap and update the badge to `v0.3`.

- [ ] **Step 2: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, output pristine.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git -c commit.gpgsign=false commit -m "docs: update README for v0.3 (Mask2Former, benchmark page)"
git push origin main
```

---

## Self-Review

- **Spec coverage:** Mask2Former wrapper (T1), benchmark helper (T2), dep + loader + e2e smoke (T3), segmentation page option (T4), benchmark page (T5), README/roadmap + push (T6). Lazy import + mocked tests enforced in T1. ResNet-50 panoptic honored. `st.scatter_chart` (no matplotlib) honored in T5. Detection-focused benchmark honored (Mask2Former absent from scatter). RF-DETR-Seg correctly absent. All spec sections covered.
- **Placeholder scan:** code steps contain full code; README edit (T6) is described by content rather than full prose — acceptable, authored against the existing README during execution. No "TODO"/"TBD" in code.
- **Type consistency:** `Prediction(boxes, labels, scores, masks, latency_ms)` reused unchanged; `Mask2FormerWrapper(model_id=...)`, `get_mask2former()`, `measure_latency(model, image, **kwargs) -> float`, `build_benchmark_rows(measured) -> list[dict]`, `PUBLISHED_MAP` keys consistent between definition (T1–T3) and use (T4–T5).
- **Known risk (T3 Step 5):** if installed `transformers` panoptic post-processing keys differ from `segmentation`/`segments_info`/`id`/`label_id`/`score`, the smoke test reports it so Task 1's wrapper + test are corrected before the page tasks rely on it.
