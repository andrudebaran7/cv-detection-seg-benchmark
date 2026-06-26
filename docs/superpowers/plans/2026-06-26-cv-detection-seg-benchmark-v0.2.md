# cv-detection-seg-benchmark v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RF-DETR (detection) and YOLO-World (open-vocabulary) wrappers and a side-by-side comparison page to the existing benchmark, all through the unified `DetectionSegModel` interface.

**Architecture:** Two new wrappers implement the existing `DetectionSegModel` ABC and return the existing `Prediction` dataclass, so the app stays model-agnostic. RF-DETR maps a `supervision.Detections` object to `Prediction`; YOLO-World takes user text classes via `predict(image, classes=[...])`. A pure `comparison.run_comparison` helper drives a new Streamlit comparison page; the detection page gains a model selector and a new open-vocabulary page is added. Heavy backends are lazy-imported; all wrapper tests mock the backend.

**Tech Stack:** Python 3.13, ultralytics (YOLOWorld), rfdetr, supervision, streamlit, numpy, pillow, pytest.

## Global Constraints

- Python >= 3.9. Use the existing project venv `.venv`.
- Repo license remains **AGPL-3.0**; new deps `rfdetr` and `supervision` are Apache-2.0.
- Default weights remain **nano/tiny**: RF-DETR uses `RFDETRNano`; YOLO-World uses `yolov8s-world.pt`.
- All wrappers MUST **lazy-import** their backend (import inside methods, never at module top).
- Tests MUST **mock** the backend — never download weights in tests.
- `Prediction` (from `models/base.py`) remains the only data contract; `predict()` accepts `**kwargs`.
- Comparison page is **detection-focused**, loads models **on demand**, caps simultaneous models at **3**.
- RF-DETR is **detection only** in v0.2 (RF-DETR-Seg deferred to v0.3).
- Commit message bodies end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Run git commits with `git -c commit.gpgsign=false`.

---

## File Structure

- `models/rfdetr_wrapper.py` (new) — `RfDetrWrapper(DetectionSegModel)`, supervision→Prediction.
- `models/yoloworld_wrapper.py` (new) — `YoloWorldWrapper(DetectionSegModel)`, open-vocab.
- `app/components/comparison.py` (new) — `ComparisonResult` + `run_comparison()` pure helper.
- `app/components/model_runner.py` (modify) — add `get_rfdetr`, `get_yoloworld`.
- `app/pages/1_Detection.py` (modify) — model selector YOLO11n / RF-DETR.
- `app/pages/3_OpenVocab.py` (new) — YOLO-World page.
- `app/pages/4_Comparison.py` (new) — side-by-side page.
- `requirements.txt`, `requirements-light.txt` (modify) — add `rfdetr`, `supervision`.
- `README.md` (modify) — model table + roadmap update.
- Tests: `tests/test_rfdetr_wrapper.py`, `tests/test_yoloworld_wrapper.py`, `tests/test_comparison.py` (new).

Work from: `/home/andru/Projects_claude/cv-detection-seg-benchmark`. Branch `main`.

---

## Task 1: RF-DETR wrapper (detection)

**Files:**
- Create: `models/rfdetr_wrapper.py`
- Test: `tests/test_rfdetr_wrapper.py`

**Interfaces:**
- Consumes: `Box`, `Prediction`, `DetectionSegModel` from `models.base`.
- Produces:
  - `class RfDetrWrapper(DetectionSegModel)`, `__init__(self, model_name: str = "nano")` storing `self.model_name` and `self._model = None`.
  - `_load(self)` — lazy-imports `rfdetr`, instantiates `rfdetr.RFDETRNano()`, caches and returns `self._model`.
  - `predict(self, image, threshold: float = 0.5, **kwargs) -> Prediction` — calls `model.predict(image, threshold=threshold)` returning a `supervision.Detections`-like object with `.xyxy` (Nx4 array), `.confidence` (N,), `.class_id` (N,); maps to `Prediction` with COCO labels, `masks=None`, measured `latency_ms`.

- [ ] **Step 1: Write the failing test (mock rfdetr + supervision-like detections)**

```python
# tests/test_rfdetr_wrapper.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np

from models.base import Prediction


def _fake_detections():
    det = MagicMock()
    det.xyxy = np.array([[5.0, 6.0, 7.0, 8.0]])
    det.confidence = np.array([0.77])
    det.class_id = np.array([0])  # COCO id 0 -> "person"
    return det


def _install_fake_rfdetr(det):
    mod = types.ModuleType("rfdetr")
    instance = MagicMock()
    instance.predict = MagicMock(return_value=det)
    mod.RFDETRNano = MagicMock(return_value=instance)
    sys.modules["rfdetr"] = mod
    return mod


def test_rfdetr_predict_maps_to_prediction():
    _install_fake_rfdetr(_fake_detections())
    from models.rfdetr_wrapper import RfDetrWrapper

    pred = RfDetrWrapper().predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert isinstance(pred, Prediction)
    assert len(pred) == 1
    assert pred.boxes[0].x1 == 5.0 and pred.boxes[0].y2 == 8.0
    assert pred.scores == [0.77]
    assert pred.labels == ["person"]
    assert pred.masks is None
    assert pred.latency_ms >= 0.0


def test_rfdetr_lazy_import():
    sys.modules.pop("rfdetr", None)
    import importlib
    import models.rfdetr_wrapper as rw
    importlib.reload(rw)  # importing must not require rfdetr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rfdetr_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.rfdetr_wrapper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/rfdetr_wrapper.py
from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction

# 80 COCO class names, index-aligned with RF-DETR class_id.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class RfDetrWrapper(DetectionSegModel):
    def __init__(self, model_name: str = "nano") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            import rfdetr  # lazy import

            self._model = rfdetr.RFDETRNano()
        return self._model

    def predict(self, image: Any, threshold: float = 0.5, **kwargs: Any) -> Prediction:
        model = self._load()
        start = time.perf_counter()
        det = model.predict(image, threshold=threshold)
        latency_ms = (time.perf_counter() - start) * 1000.0

        boxes = [Box(float(x1), float(y1), float(x2), float(y2))
                 for x1, y1, x2, y2 in det.xyxy]
        scores = [float(c) for c in det.confidence]
        labels = [COCO_CLASSES[int(i)] if 0 <= int(i) < len(COCO_CLASSES) else str(int(i))
                  for i in det.class_id]

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rfdetr_wrapper.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/rfdetr_wrapper.py tests/test_rfdetr_wrapper.py
git -c commit.gpgsign=false commit -m "feat(models): add RfDetrWrapper (detection) with mocked tests"
```

---

## Task 2: YOLO-World wrapper (open-vocabulary)

**Files:**
- Create: `models/yoloworld_wrapper.py`
- Test: `tests/test_yoloworld_wrapper.py`

**Interfaces:**
- Consumes: `Box`, `Prediction`, `DetectionSegModel` from `models.base`.
- Produces:
  - `class YoloWorldWrapper(DetectionSegModel)`, `__init__(self, weights: str = "yolov8s-world.pt")`.
  - `_load(self)` — lazy-imports `ultralytics.YOLOWorld`, caches `self._model`.
  - `predict(self, image, classes: list[str] | None = None, conf: float = 0.25, **kwargs) -> Prediction` — raises `ValueError` if `classes` is falsy; calls `model.set_classes(classes)` then `model(image, conf=conf, verbose=False)`; maps result[0] (`boxes.xyxy/conf/cls`, `names`) to `Prediction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yoloworld_wrapper.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from models.base import Prediction


def _make_fake_result():
    r = MagicMock()
    r.names = {0: "cat", 1: "umbrella"}
    r.boxes.xyxy.tolist.return_value = [[1.0, 2.0, 3.0, 4.0]]
    r.boxes.conf.tolist.return_value = [0.66]
    r.boxes.cls.tolist.return_value = [1]
    r.masks = None
    return r


def _install_fake_yoloworld(fake_result):
    mod = types.ModuleType("ultralytics")
    instance = MagicMock(return_value=[fake_result])
    instance.set_classes = MagicMock()
    mod.YOLOWorld = MagicMock(return_value=instance)
    sys.modules["ultralytics"] = mod
    return instance


def test_yoloworld_sets_classes_and_maps_prediction():
    instance = _install_fake_yoloworld(_make_fake_result())
    from models.yoloworld_wrapper import YoloWorldWrapper

    pred = YoloWorldWrapper().predict(
        np.zeros((32, 32, 3), dtype=np.uint8), classes=["cat", "umbrella"]
    )

    instance.set_classes.assert_called_once_with(["cat", "umbrella"])
    assert isinstance(pred, Prediction)
    assert pred.labels == ["umbrella"]
    assert pred.scores == [0.66]
    assert pred.boxes[0].x1 == 1.0
    assert pred.latency_ms >= 0.0


def test_yoloworld_requires_classes():
    _install_fake_yoloworld(_make_fake_result())
    from models.yoloworld_wrapper import YoloWorldWrapper

    with pytest.raises(ValueError):
        YoloWorldWrapper().predict(np.zeros((32, 32, 3), dtype=np.uint8), classes=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yoloworld_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.yoloworld_wrapper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/yoloworld_wrapper.py
from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction


class YoloWorldWrapper(DetectionSegModel):
    def __init__(self, weights: str = "yolov8s-world.pt") -> None:
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLOWorld  # lazy import

            self._model = YOLOWorld(self.weights)
        return self._model

    def predict(
        self,
        image: Any,
        classes: list[str] | None = None,
        conf: float = 0.25,
        **kwargs: Any,
    ) -> Prediction:
        if not classes:
            raise ValueError("YOLO-World requires at least one text class.")

        model = self._load()
        model.set_classes(classes)

        start = time.perf_counter()
        results = model(image, conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        xyxy = result.boxes.xyxy.tolist()
        confs = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()
        names = result.names

        boxes = [Box(*coords) for coords in xyxy]
        labels = [names[int(c)] for c in cls_ids]
        scores = [float(s) for s in confs]

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yoloworld_wrapper.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/yoloworld_wrapper.py tests/test_yoloworld_wrapper.py
git -c commit.gpgsign=false commit -m "feat(models): add YoloWorldWrapper (open-vocab) with mocked tests"
```

---

## Task 3: Comparison helper

**Files:**
- Create: `app/components/comparison.py`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `DetectionSegModel`, `Prediction` from `models.base`; `draw_prediction` from `app.components.visualization`.
- Produces:
  - `@dataclass ComparisonResult(name: str, prediction: Prediction, image: "PIL.Image.Image")`.
  - `def run_comparison(image, models: dict[str, DetectionSegModel], per_model_kwargs: dict[str, dict] | None = None) -> list[ComparisonResult]` — runs each model in dict order on the same image, renders each via `draw_prediction`, returns one `ComparisonResult` per model preserving order. `per_model_kwargs` maps a model name to kwargs passed to that model's `predict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comparison.py
from unittest.mock import MagicMock

from PIL import Image

from models.base import Box, Prediction
from app.components.comparison import run_comparison, ComparisonResult


def _model_returning(pred):
    m = MagicMock()
    m.predict = MagicMock(return_value=pred)
    return m


def test_run_comparison_preserves_order_and_renders():
    img = Image.new("RGB", (40, 30), "white")
    pred_a = Prediction(boxes=[Box(1, 1, 5, 5)], labels=["a"], scores=[0.9], latency_ms=3.0)
    pred_b = Prediction(boxes=[], labels=[], scores=[], latency_ms=1.0)

    models = {"A": _model_returning(pred_a), "B": _model_returning(pred_b)}
    results = run_comparison(img, models)

    assert [r.name for r in results] == ["A", "B"]
    assert all(isinstance(r, ComparisonResult) for r in results)
    assert results[0].prediction is pred_a
    assert isinstance(results[0].image, Image.Image)
    assert results[0].image.size == (40, 30)


def test_run_comparison_passes_per_model_kwargs():
    img = Image.new("RGB", (10, 10), "white")
    pred = Prediction(boxes=[], labels=[], scores=[], latency_ms=0.0)
    model = _model_returning(pred)

    run_comparison(img, {"W": model}, per_model_kwargs={"W": {"classes": ["dog"]}})

    model.predict.assert_called_once()
    _, kwargs = model.predict.call_args
    assert kwargs.get("classes") == ["dog"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_comparison.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.components.comparison'`.

- [ ] **Step 3: Write minimal implementation**

```python
# app/components/comparison.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.base import DetectionSegModel, Prediction
from app.components.visualization import draw_prediction


@dataclass
class ComparisonResult:
    name: str
    prediction: Prediction
    image: Any  # PIL.Image.Image


def run_comparison(
    image: Any,
    models: "dict[str, DetectionSegModel]",
    per_model_kwargs: "dict[str, dict] | None" = None,
) -> "list[ComparisonResult]":
    per_model_kwargs = per_model_kwargs or {}
    results: list[ComparisonResult] = []
    for name, model in models.items():
        kwargs = per_model_kwargs.get(name, {})
        prediction = model.predict(image, **kwargs)
        rendered = draw_prediction(image, prediction)
        results.append(ComparisonResult(name=name, prediction=prediction, image=rendered))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_comparison.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (v0.1 + the three new test files), output pristine.

- [ ] **Step 6: Commit**

```bash
git add app/components/comparison.py tests/test_comparison.py
git -c commit.gpgsign=false commit -m "feat(app): add run_comparison helper and ComparisonResult"
```

---

## Task 4: Dependencies + model_runner loaders

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-light.txt`
- Modify: `app/components/model_runner.py`

**Interfaces:**
- Consumes: `RfDetrWrapper`, `YoloWorldWrapper`.
- Produces: `get_rfdetr() -> RfDetrWrapper`, `get_yoloworld() -> YoloWorldWrapper` (both `@st.cache_resource`).

This task installs the new backends and is verified by an end-to-end smoke test (no new unit test).

- [ ] **Step 1: Add deps to `requirements.txt`** (append after `pillow>=10.0`)

```
rfdetr>=1.8
supervision>=0.20
```

- [ ] **Step 2: Add deps to `requirements-light.txt`** (append after `pillow>=10.0`)

```
rfdetr>=1.8
supervision>=0.20
```

- [ ] **Step 3: Install the new backends**

Run: `.venv/bin/pip install rfdetr supervision`
Expected: both install successfully.

- [ ] **Step 4: Add loaders to `app/components/model_runner.py`** (append these functions)

```python
from models.rfdetr_wrapper import RfDetrWrapper
from models.yoloworld_wrapper import YoloWorldWrapper


@st.cache_resource
def get_rfdetr(model_name: str = "nano") -> RfDetrWrapper:
    return RfDetrWrapper(model_name)


@st.cache_resource
def get_yoloworld(weights: str = "yolov8s-world.pt") -> YoloWorldWrapper:
    return YoloWorldWrapper(weights)
```

- [ ] **Step 5: End-to-end RF-DETR smoke test (real weights)**

```bash
.venv/bin/python - <<'PY'
import numpy as np
from PIL import Image
from models.rfdetr_wrapper import RfDetrWrapper

img = np.array(Image.open("data/sample_images/bus.jpg").convert("RGB"))
pred = RfDetrWrapper().predict(img)
print("TYPE", type(pred).__name__, "N", len(pred))
print("LABELS", sorted(set(pred.labels))[:5])
print("SCORES_OK", all(0 <= s <= 1 for s in pred.scores))
PY
```

Expected: prints a `Prediction` with one or more detections (e.g. labels include `bus`/`person`) and `SCORES_OK True`. If `RfDetrWrapper`'s `det.xyxy/.confidence/.class_id` attribute names do not match the installed `rfdetr` output, report `DONE_WITH_CONCERNS` with the actual attribute names so the wrapper (Task 1) can be corrected and its test updated.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-light.txt app/components/model_runner.py
git -c commit.gpgsign=false commit -m "feat(app): add rfdetr/supervision deps and model_runner loaders"
```

---

## Task 5: Detection page model selector

**Files:**
- Modify: `app/pages/1_Detection.py`

**Interfaces:**
- Consumes: `get_yolo`, `get_rfdetr` from `app.components.model_runner`; `draw_prediction`.

Verified by launching the app (manual smoke test).

- [ ] **Step 1: Rewrite `app/pages/1_Detection.py`**

```python
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr
from app.components.visualization import draw_prediction

st.title("Detection")

model_name = st.selectbox("Model", ["YOLO11n", "RF-DETR-nano"])
conf = st.slider("Confidence / threshold", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)
    with st.spinner(f"Running {model_name}..."):
        if model_name == "YOLO11n":
            pred = get_yolo().predict(arr, conf=conf)
        else:
            pred = get_rfdetr().predict(arr, threshold=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"labels": pred.labels, "scores": [round(s, 3) for s in pred.scores]})
```

- [ ] **Step 2: Smoke-test the page**

```bash
.venv/bin/streamlit run app/main.py --server.headless true --server.port 8766 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8766
```

Expected: `200`; then stop the server. Manually confirm the Detection page shows the model dropdown.

- [ ] **Step 3: Commit**

```bash
git add app/pages/1_Detection.py
git -c commit.gpgsign=false commit -m "feat(app): add model selector (YOLO11n/RF-DETR) to detection page"
```

---

## Task 6: Open-Vocabulary page (YOLO-World)

**Files:**
- Create: `app/pages/3_OpenVocab.py`

**Interfaces:**
- Consumes: `get_yoloworld` from `app.components.model_runner`; `draw_prediction`.

Verified by launching the app (manual smoke test).

- [ ] **Step 1: Write `app/pages/3_OpenVocab.py`**

```python
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yoloworld
from app.components.visualization import draw_prediction

st.title("Open-Vocabulary Detection — YOLO-World")

COMMON = ["person", "car", "dog", "cat", "bottle", "chair", "laptop", "cell phone",
          "traffic light", "backpack"]

selected = st.multiselect("Common classes", COMMON, default=["person"])
extra = st.text_input("Extra classes (comma-separated)", "")
classes = selected + [c.strip() for c in extra.split(",") if c.strip()]

conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None and not classes:
    st.warning("Select or type at least one class to detect.")
elif file is not None:
    image = Image.open(file).convert("RGB")
    with st.spinner("Running YOLO-World..."):
        pred = get_yoloworld().predict(np.array(image), classes=classes, conf=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"classes": classes, "labels": pred.labels})
```

- [ ] **Step 2: Smoke-test**

```bash
.venv/bin/streamlit run app/main.py --server.headless true --server.port 8767 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8767
```

Expected: `200`; stop the server. Confirm the Open-Vocabulary page appears in the sidebar.

- [ ] **Step 3: Commit**

```bash
git add app/pages/3_OpenVocab.py
git -c commit.gpgsign=false commit -m "feat(app): add open-vocabulary page (YOLO-World)"
```

---

## Task 7: Comparison page

**Files:**
- Create: `app/pages/4_Comparison.py`

**Interfaces:**
- Consumes: `get_yolo`, `get_rfdetr`, `get_yoloworld`; `run_comparison`, `ComparisonResult`.

Verified by launching the app (manual smoke test).

- [ ] **Step 1: Write `app/pages/4_Comparison.py`**

```python
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo, get_rfdetr, get_yoloworld
from app.components.comparison import run_comparison

st.title("Comparison")

ALL = ["YOLO11n", "RF-DETR-nano", "YOLO-World"]
chosen = st.multiselect("Detectors (pick 2–3)", ALL, default=["YOLO11n", "RF-DETR-nano"])
wc_classes = st.text_input("YOLO-World classes (comma-separated)", "person, car")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None and not (2 <= len(chosen) <= 3):
    st.warning("Pick between 2 and 3 detectors.")
elif file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)

    loaders = {"YOLO11n": get_yolo, "RF-DETR-nano": get_rfdetr, "YOLO-World": get_yoloworld}
    models = {name: loaders[name]() for name in chosen}
    per_kwargs = {}
    if "YOLO-World" in models:
        per_kwargs["YOLO-World"] = {
            "classes": [c.strip() for c in wc_classes.split(",") if c.strip()]
        }

    with st.spinner("Running models..."):
        results = run_comparison(arr, models, per_model_kwargs=per_kwargs)

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        col.image(r.image, caption=r.name)
    st.table(
        {
            "model": [r.name for r in results],
            "objects": [len(r.prediction) for r in results],
            "latency_ms": [round(r.prediction.latency_ms) for r in results],
        }
    )
```

- [ ] **Step 2: Smoke-test**

```bash
.venv/bin/streamlit run app/main.py --server.headless true --server.port 8768 &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8768
```

Expected: `200`; stop the server. Confirm the Comparison page renders the multiselect.

- [ ] **Step 3: Commit**

```bash
git add app/pages/4_Comparison.py
git -c commit.gpgsign=false commit -m "feat(app): add side-by-side comparison page"
```

---

## Task 8: README + roadmap update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the model table** to add rows for `RF-DETR-nano` (detection, `rfdetr`) and `YOLO-World` (open-vocab, `ultralytics`); add a short "Pages" subsection listing Detection, Segmentation, Open-Vocabulary, Comparison; move v0.2 items from roadmap into the shipped list and mark v0.2 done.

- [ ] **Step 2: Run the full suite one more time**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, output pristine.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git -c commit.gpgsign=false commit -m "docs: update README for v0.2 (RF-DETR, YOLO-World, comparison)"
git push origin main
```

---

## Self-Review

- **Spec coverage:** RF-DETR wrapper (T1), YOLO-World wrapper (T2), comparison helper (T3), deps + loaders (T4), detection selector (T5), open-vocab page (T6), comparison page (T7), README/roadmap (T8). Lazy imports + mocked tests enforced in T1/T2. RF-DETR detection-only honored. `classes` via `**kwargs` honored. Comparison detection-focused, on-demand load, 2–3 cap honored in T7. All spec sections covered.
- **Placeholder scan:** code steps contain full code; README edit (T8) is described by content rather than full prose — acceptable, as exact wording is authored during execution against the existing README. No "TODO"/"TBD" in code.
- **Type consistency:** `Prediction(boxes, labels, scores, masks, latency_ms)` and `Box(x1,y1,x2,y2)` reused unchanged; `RfDetrWrapper.predict(image, threshold=...)`, `YoloWorldWrapper.predict(image, classes=..., conf=...)`, `run_comparison(image, models, per_model_kwargs=...)`, `ComparisonResult(name, prediction, image)`, `get_rfdetr`/`get_yoloworld` consistent between definition (T1–T4) and use (T5–T7).
- **Known risk (T4 Step 5):** if installed `rfdetr` output attribute names differ from `.xyxy/.confidence/.class_id`, the smoke test reports it so Task 1's wrapper + test are corrected before the page tasks rely on it.
