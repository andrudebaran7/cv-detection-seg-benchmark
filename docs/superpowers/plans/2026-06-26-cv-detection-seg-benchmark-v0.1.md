# cv-detection-seg-benchmark v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a deployable Streamlit app that runs object detection (YOLO) and segmentation (YOLO-seg + SAM 2) on an uploaded image through a unified model interface, plus a private companion LaTeX report — both pushed to GitHub.

**Architecture:** A `models/base.py` defines a `Prediction` dataclass and a `DetectionSegModel` ABC; concrete wrappers (`YoloWrapper`, `Sam2Wrapper`) adapt `ultralytics` output to `Prediction`. The Streamlit app only consumes `predict()` and renders the result via a shared visualization helper. Heavy deps (`torch`/`ultralytics`) are lazy-imported so the test suite runs with mocks and no weight downloads. A separate repo holds the LaTeX report condensed from the source docx.

**Tech Stack:** Python 3.13, ultralytics, torch, streamlit, numpy, pillow, pytest; LaTeX (article class) + bibtex.

## Global Constraints

- Python >= 3.9 (developed on 3.13). Use the project venv at `cv-detection-seg-benchmark/.venv`.
- Repo A license: **AGPL-3.0** (because `ultralytics`/`yolov12` are AGPL-3.0).
- Default model weights are **nano/tiny only**: `yolo11n.pt`, `yolo11n-seg.pt`, `sam2.1_t.pt` (Streamlit Community Cloud ~1 GB RAM limit).
- Wrappers MUST lazy-import `ultralytics` (import inside methods, never at module top level) so tests run without torch.
- Tests MUST mock the underlying model — never download weights in tests.
- `Prediction` dataclass is the only data contract between models and the app.
- Repo A `cv-detection-seg-benchmark` is **public**; Repo B `cv-detection-seg-report` is **private**.
- GitHub repo creation is done by the user (empty, no README/license/gitignore); Claude only adds remotes and pushes.

---

## File Structure

Repo A — `cv-detection-seg-benchmark/`:
- `models/base.py` — `Box`, `Prediction` dataclasses; `DetectionSegModel` ABC.
- `models/yolo_wrapper.py` — `YoloWrapper(DetectionSegModel)` for detection + instance seg.
- `models/sam2_wrapper.py` — `Sam2Wrapper(DetectionSegModel)` for point/box-prompted seg.
- `app/components/visualization.py` — `draw_prediction(image, prediction) -> PIL.Image`.
- `app/components/model_runner.py` — cached model loaders (`@st.cache_resource`).
- `app/main.py` — Streamlit home page.
- `app/pages/1_Detection.py`, `app/pages/2_Segmentation.py` — task pages.
- `tests/test_base.py`, `tests/test_yolo_wrapper.py`, `tests/test_sam2_wrapper.py`, `tests/test_visualization.py`.
- `requirements.txt`, `requirements-light.txt`, `.gitignore`, `LICENSE`, `README.md`.

Repo B — `cv-detection-seg-report/`:
- `main.tex`, `references.bib`, `sections/01..05*.tex`, `figures/`, `.gitignore`, `README.md`.

---

## Task 1: Repo A scaffolding, venv, and tooling

**Files:**
- Create: `cv-detection-seg-benchmark/.gitignore`
- Create: `cv-detection-seg-benchmark/requirements.txt`
- Create: `cv-detection-seg-benchmark/requirements-light.txt`
- Create: `cv-detection-seg-benchmark/LICENSE`
- Create: `cv-detection-seg-benchmark/README.md`
- Create: `cv-detection-seg-benchmark/models/__init__.py`, `cv-detection-seg-benchmark/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a git repo with a working `.venv` and `pytest` runnable.

- [ ] **Step 1: Initialize git repo and directory layout**

```bash
cd /home/andru/Projects_claude/cv-detection-seg-benchmark
git init
mkdir -p models app/pages app/components data/sample_images tests figures
touch models/__init__.py tests/__init__.py app/components/__init__.py
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.pt
*.onnx
.streamlit/secrets.toml
.DS_Store
data/sample_images/*.tmp
```

- [ ] **Step 3: Write `requirements.txt`**

```
ultralytics>=8.3
torch>=2.2
streamlit>=1.40
numpy>=1.26
pillow>=10.0
pytest>=8.0
```

- [ ] **Step 4: Write `requirements-light.txt`** (Streamlit Cloud; CPU torch)

```
--extra-index-url https://download.pytorch.org/whl/cpu
ultralytics>=8.3
torch>=2.2
streamlit>=1.40
numpy>=1.26
pillow>=10.0
```

- [ ] **Step 5: Add AGPL-3.0 LICENSE**

Run: `curl -sL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE` and verify the first line contains "GNU AFFERO GENERAL PUBLIC LICENSE".

- [ ] **Step 6: Create venv and install test-only deps for now**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip install pytest numpy pillow
```

(Heavy deps `torch`/`ultralytics`/`streamlit` are installed in Task 6 / at app-run time; tests don't need them.)

- [ ] **Step 7: Write minimal `README.md`**

```markdown
# cv-detection-seg-benchmark

Interactive comparison of object detection and image segmentation models
(YOLO11, SAM 2) with a Streamlit demo. Milestone: v0.1.

## Quickstart
\`\`\`bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/main.py
\`\`\`

## Tests
\`\`\`bash
.venv/bin/python -m pytest -v
\`\`\`

License: AGPL-3.0.
```

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: scaffold repo, venv, deps, AGPL license"
```

---

## Task 2: `models/base.py` — data contract + ABC

**Files:**
- Create: `cv-detection-seg-benchmark/models/base.py`
- Test: `cv-detection-seg-benchmark/tests/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass Box(x1: float, y1: float, x2: float, y2: float)`
  - `@dataclass Prediction(boxes: list[Box], labels: list[str], scores: list[float], masks: list | None = None, latency_ms: float = 0.0)` with `def __len__(self) -> int` returning `len(self.boxes)`.
  - `class DetectionSegModel(ABC)` with `@abstractmethod def predict(self, image, **kwargs) -> Prediction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base.py
import pytest
from models.base import Box, Prediction, DetectionSegModel


def test_prediction_len_counts_boxes():
    p = Prediction(
        boxes=[Box(0, 0, 1, 1), Box(2, 2, 3, 3)],
        labels=["cat", "dog"],
        scores=[0.9, 0.8],
        latency_ms=12.5,
    )
    assert len(p) == 2
    assert p.masks is None
    assert p.latency_ms == 12.5


def test_detectionsegmodel_is_abstract():
    with pytest.raises(TypeError):
        DetectionSegModel()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.base'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Prediction:
    boxes: list[Box]
    labels: list[str]
    scores: list[float]
    masks: list[Any] | None = None
    latency_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.boxes)


class DetectionSegModel(ABC):
    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Prediction:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_base.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/base.py tests/test_base.py
git commit -m "feat(models): add Prediction dataclass and DetectionSegModel ABC"
```

---

## Task 3: `models/yolo_wrapper.py` — YOLO detection + segmentation

**Files:**
- Create: `cv-detection-seg-benchmark/models/yolo_wrapper.py`
- Test: `cv-detection-seg-benchmark/tests/test_yolo_wrapper.py`

**Interfaces:**
- Consumes: `Box`, `Prediction`, `DetectionSegModel` from `models.base`.
- Produces:
  - `class YoloWrapper(DetectionSegModel)`, constructor `__init__(self, weights: str = "yolo11n.pt")` storing `self.weights` and `self._model = None`.
  - `def _load(self)` — lazy-imports `ultralytics.YOLO`, sets and returns `self._model` (cached).
  - `def predict(self, image, conf: float = 0.25, **kwargs) -> Prediction` — runs the model, maps the first result to `Prediction` (boxes from `result.boxes.xyxy`, labels via `result.names[cls]`, scores from `result.boxes.conf`, masks from `result.masks.data` if present, `latency_ms` measured with `time.perf_counter`).

- [ ] **Step 1: Write the failing test (mock ultralytics, no download)**

```python
# tests/test_yolo_wrapper.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np

from models.base import Prediction


def _install_fake_ultralytics(fake_result):
    """Inject a fake `ultralytics` module exposing YOLO -> callable returning [fake_result]."""
    mod = types.ModuleType("ultralytics")
    model_instance = MagicMock(return_value=[fake_result])
    mod.YOLO = MagicMock(return_value=model_instance)
    sys.modules["ultralytics"] = mod
    return mod


def _make_fake_result():
    r = MagicMock()
    r.names = {0: "person", 16: "dog"}
    r.boxes.xyxy.tolist.return_value = [[10.0, 20.0, 30.0, 40.0]]
    r.boxes.conf.tolist.return_value = [0.91]
    r.boxes.cls.tolist.return_value = [0]
    r.masks = None
    return r


def test_yolo_predict_maps_boxes_labels_scores(monkeypatch):
    _install_fake_ultralytics(_make_fake_result())
    from models.yolo_wrapper import YoloWrapper

    pred = YoloWrapper().predict(np.zeros((64, 64, 3), dtype=np.uint8))

    assert isinstance(pred, Prediction)
    assert len(pred) == 1
    assert pred.labels == ["person"]
    assert pred.scores == [0.91]
    assert pred.boxes[0].x1 == 10.0 and pred.boxes[0].y2 == 40.0
    assert pred.latency_ms >= 0.0


def test_yolo_lazy_import_not_at_module_top():
    # Importing the wrapper module must not require ultralytics.
    sys.modules.pop("ultralytics", None)
    import importlib
    import models.yolo_wrapper as yw
    importlib.reload(yw)  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_yolo_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.yolo_wrapper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/yolo_wrapper.py
from __future__ import annotations

import time
from typing import Any

from models.base import Box, DetectionSegModel, Prediction


class YoloWrapper(DetectionSegModel):
    def __init__(self, weights: str = "yolo11n.pt") -> None:
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy import

            self._model = YOLO(self.weights)
        return self._model

    def predict(self, image: Any, conf: float = 0.25, **kwargs: Any) -> Prediction:
        model = self._load()
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
        masks = result.masks.data.tolist() if result.masks is not None else None

        return Prediction(
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=masks,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_yolo_wrapper.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/yolo_wrapper.py tests/test_yolo_wrapper.py
git commit -m "feat(models): add YoloWrapper with lazy import and mocked tests"
```

---

## Task 4: `models/sam2_wrapper.py` — SAM 2 promptable segmentation

**Files:**
- Create: `cv-detection-seg-benchmark/models/sam2_wrapper.py`
- Test: `cv-detection-seg-benchmark/tests/test_sam2_wrapper.py`

**Interfaces:**
- Consumes: `Prediction`, `DetectionSegModel` from `models.base`.
- Produces:
  - `class Sam2Wrapper(DetectionSegModel)`, `__init__(self, weights: str = "sam2.1_t.pt")`.
  - `def _load(self)` — lazy-imports `ultralytics.SAM`, caches `self._model`.
  - `def predict(self, image, points=None, bboxes=None, **kwargs) -> Prediction` — calls `model(image, points=points, bboxes=bboxes)`, maps `result.masks.data` to `Prediction.masks`; `boxes`/`labels`/`scores` are empty lists (SAM is promptable, not a classifier). Raises `ValueError` if both `points` and `bboxes` are `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam2_wrapper.py
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from models.base import Prediction


def _install_fake_ultralytics_sam(fake_result):
    mod = types.ModuleType("ultralytics")
    model_instance = MagicMock(return_value=[fake_result])
    mod.SAM = MagicMock(return_value=model_instance)
    sys.modules["ultralytics"] = mod


def _make_fake_sam_result():
    r = MagicMock()
    r.masks.data.tolist.return_value = [[[0, 1], [1, 0]]]
    return r


def test_sam2_predict_returns_masks(monkeypatch):
    _install_fake_ultralytics_sam(_make_fake_sam_result())
    from models.sam2_wrapper import Sam2Wrapper

    pred = Sam2Wrapper().predict(
        np.zeros((64, 64, 3), dtype=np.uint8), points=[[32, 32]]
    )
    assert isinstance(pred, Prediction)
    assert pred.masks == [[[0, 1], [1, 0]]]
    assert pred.boxes == []
    assert pred.latency_ms >= 0.0


def test_sam2_requires_a_prompt():
    _install_fake_ultralytics_sam(_make_fake_sam_result())
    from models.sam2_wrapper import Sam2Wrapper

    with pytest.raises(ValueError):
        Sam2Wrapper().predict(np.zeros((64, 64, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sam2_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.sam2_wrapper'`.

- [ ] **Step 3: Write minimal implementation**

```python
# models/sam2_wrapper.py
from __future__ import annotations

import time
from typing import Any

from models.base import DetectionSegModel, Prediction


class Sam2Wrapper(DetectionSegModel):
    def __init__(self, weights: str = "sam2.1_t.pt") -> None:
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM  # lazy import

            self._model = SAM(self.weights)
        return self._model

    def predict(
        self,
        image: Any,
        points: Any = None,
        bboxes: Any = None,
        **kwargs: Any,
    ) -> Prediction:
        if points is None and bboxes is None:
            raise ValueError("SAM 2 requires a prompt: pass `points` or `bboxes`.")

        model = self._load()
        start = time.perf_counter()
        results = model(image, points=points, bboxes=bboxes, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        masks = result.masks.data.tolist() if result.masks is not None else None

        return Prediction(
            boxes=[],
            labels=[],
            scores=[],
            masks=masks,
            latency_ms=latency_ms,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sam2_wrapper.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add models/sam2_wrapper.py tests/test_sam2_wrapper.py
git commit -m "feat(models): add Sam2Wrapper (promptable) with mocked tests"
```

---

## Task 5: `app/components/visualization.py` — draw predictions

**Files:**
- Create: `cv-detection-seg-benchmark/app/components/visualization.py`
- Test: `cv-detection-seg-benchmark/tests/test_visualization.py`

**Interfaces:**
- Consumes: `Box`, `Prediction` from `models.base`; `PIL.Image`.
- Produces:
  - `def draw_prediction(image: "PIL.Image.Image", prediction: Prediction) -> "PIL.Image.Image"` — returns a NEW image (does not mutate input) with rectangles for each box and `label score` text; mask overlay drawn when `prediction.masks` is not None. Uses only `pillow` and `numpy` (no torch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visualization.py
from PIL import Image

from models.base import Box, Prediction
from app.components.visualization import draw_prediction


def test_draw_prediction_returns_new_image_same_size():
    img = Image.new("RGB", (100, 80), "white")
    pred = Prediction(
        boxes=[Box(10, 10, 50, 50)],
        labels=["cat"],
        scores=[0.88],
        latency_ms=5.0,
    )
    out = draw_prediction(img, pred)

    assert isinstance(out, Image.Image)
    assert out.size == (100, 80)
    assert out is not img  # must not mutate the original
    # something was drawn: output differs from a blank white image
    assert list(out.getdata()) != list(img.getdata())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_visualization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.components.visualization'`.

(If `app` is not importable, add empty `app/__init__.py` and `app/components/__init__.py`; `tests/` runs from repo root.)

- [ ] **Step 3: Write minimal implementation**

```python
# app/components/visualization.py
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from models.base import Prediction

_BOX_COLOR = (255, 0, 0)
_MASK_COLOR = np.array([0, 200, 255], dtype=np.uint8)


def draw_prediction(image: "Image.Image", prediction: Prediction) -> "Image.Image":
    out = image.convert("RGB").copy()

    if prediction.masks:
        arr = np.array(out)
        for mask in prediction.masks:
            m = np.array(mask)
            if m.shape[:2] != arr.shape[:2]:
                m = np.array(
                    Image.fromarray((np.array(mask) * 255).astype("uint8"))
                    .resize((arr.shape[1], arr.shape[0]))
                ) > 127
            sel = m.astype(bool)
            arr[sel] = (0.5 * arr[sel] + 0.5 * _MASK_COLOR).astype(np.uint8)
        out = Image.fromarray(arr)

    draw = ImageDraw.Draw(out)
    for box, label, score in zip(prediction.boxes, prediction.labels, prediction.scores):
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=_BOX_COLOR, width=2)
        draw.text((box.x1, max(0, box.y1 - 10)), f"{label} {score:.2f}", fill=_BOX_COLOR)

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_visualization.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass (base + yolo + sam2 + visualization).

- [ ] **Step 6: Commit**

```bash
git add app/components/visualization.py tests/test_visualization.py app/__init__.py app/components/__init__.py
git commit -m "feat(app): add draw_prediction visualization helper"
```

---

## Task 6: Streamlit app (home + detection + segmentation pages)

**Files:**
- Create: `cv-detection-seg-benchmark/app/components/model_runner.py`
- Create: `cv-detection-seg-benchmark/app/main.py`
- Create: `cv-detection-seg-benchmark/app/pages/1_Detection.py`
- Create: `cv-detection-seg-benchmark/app/pages/2_Segmentation.py`

**Interfaces:**
- Consumes: `YoloWrapper`, `Sam2Wrapper`, `draw_prediction`.
- Produces:
  - `model_runner.get_yolo(weights="yolo11n.pt") -> YoloWrapper` and `get_yolo_seg() -> YoloWrapper` and `get_sam2() -> Sam2Wrapper`, each decorated with `@st.cache_resource`.

This task installs heavy deps and is verified by launching the app (manual smoke test), not by unit tests.

- [ ] **Step 1: Install full dependencies**

```bash
cd /home/andru/Projects_claude/cv-detection-seg-benchmark
.venv/bin/pip install -r requirements.txt
```

Expected: ultralytics, torch, streamlit install successfully.

- [ ] **Step 2: Write `model_runner.py`**

```python
# app/components/model_runner.py
import streamlit as st

from models.yolo_wrapper import YoloWrapper
from models.sam2_wrapper import Sam2Wrapper


@st.cache_resource
def get_yolo(weights: str = "yolo11n.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_yolo_seg(weights: str = "yolo11n-seg.pt") -> YoloWrapper:
    return YoloWrapper(weights)


@st.cache_resource
def get_sam2(weights: str = "sam2.1_t.pt") -> Sam2Wrapper:
    return Sam2Wrapper(weights)
```

- [ ] **Step 3: Write `app/main.py`**

```python
# app/main.py
import streamlit as st

st.set_page_config(page_title="CV Detection & Segmentation Benchmark", layout="wide")

st.title("CV Detection & Segmentation Benchmark")
st.markdown(
    """
    Interactive comparison of detection and segmentation models (v0.1).

    - **Detection** — YOLO11 nano on an uploaded image.
    - **Segmentation** — YOLO11-seg (automatic) and SAM 2 tiny (point-prompted).

    Use the sidebar to pick a task. Models are nano/tiny to fit Streamlit Cloud.
    """
)
```

- [ ] **Step 4: Write `app/pages/1_Detection.py`**

```python
# app/pages/1_Detection.py
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo
from app.components.visualization import draw_prediction

st.title("Detection — YOLO11n")

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    with st.spinner("Running YOLO11n..."):
        pred = get_yolo().predict(np.array(image), conf=conf)
    st.image(draw_prediction(image, pred), caption=f"{len(pred)} objects · {pred.latency_ms:.0f} ms")
    st.write({"labels": pred.labels, "scores": [round(s, 3) for s in pred.scores]})
```

- [ ] **Step 5: Write `app/pages/2_Segmentation.py`**

```python
# app/pages/2_Segmentation.py
import numpy as np
import streamlit as st
from PIL import Image

from app.components.model_runner import get_yolo_seg, get_sam2
from app.components.visualization import draw_prediction

st.title("Segmentation — YOLO11n-seg / SAM 2")

mode = st.radio("Model", ["YOLO11n-seg (automatic)", "SAM 2 tiny (point prompt)"])
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    arr = np.array(image)

    if mode.startswith("YOLO"):
        with st.spinner("Running YOLO11n-seg..."):
            pred = get_yolo_seg().predict(arr)
    else:
        st.caption("Click point is the image center in v0.1 (interactive click lands in v0.2).")
        h, w = arr.shape[:2]
        with st.spinner("Running SAM 2 tiny..."):
            pred = get_sam2().predict(arr, points=[[w // 2, h // 2]])

    st.image(draw_prediction(image, pred), caption=f"{pred.latency_ms:.0f} ms")
```

- [ ] **Step 6: Smoke-test the app**

```bash
.venv/bin/streamlit run app/main.py --server.headless true
```

Open the URL, upload a sample image on the Detection page, confirm boxes render and
weights download on first run. Stop the server (Ctrl-C).

- [ ] **Step 7: Add 2 sample images**

Place 2 royalty-free `.jpg` images in `data/sample_images/` (e.g. from `ultralytics`'
bundled `bus.jpg`/`zidane.jpg`, which are permissively usable for demos). Verify they load.

- [ ] **Step 8: Commit**

```bash
git add app/ data/sample_images/
git commit -m "feat(app): add Streamlit home, detection, and segmentation pages"
```

---

## Task 7: Repo A README polish + push to GitHub

**Files:**
- Modify: `cv-detection-seg-benchmark/README.md`

**Interfaces:**
- Consumes: a user-created empty PUBLIC repo `git@github.com:andrudebaran7/cv-detection-seg-benchmark.git`.
- Produces: pushed `main` branch.

- [ ] **Step 1: Expand README** with model table (YOLO11n, YOLO11n-seg, SAM 2 tiny), deploy notes (`requirements-light.txt` for Streamlit Cloud), and a link to the report repo. Include the test command and the AGPL-3.0 note.

- [ ] **Step 2: Commit README**

```bash
git add README.md && git commit -m "docs: expand README with model table and deploy notes"
```

- [ ] **Step 3: PAUSE — request the empty repo**

Ask the user to create an **empty public** repo named `cv-detection-seg-benchmark`
(no README/license/gitignore) at github.com. Do not proceed until confirmed.

- [ ] **Step 4: Add remote and push**

```bash
git branch -M main
git remote add origin git@github.com:andrudebaran7/cv-detection-seg-benchmark.git
git push -u origin main
```

Expected: push succeeds; `git ls-remote origin` lists `refs/heads/main`.

---

## Task 8: Repo B — LaTeX report (condensed from docx)

**Files:**
- Create: `cv-detection-seg-report/main.tex`
- Create: `cv-detection-seg-report/references.bib`
- Create: `cv-detection-seg-report/sections/01-introduction.tex` … `05-benchmark-design.tex`
- Create: `cv-detection-seg-report/.gitignore`, `cv-detection-seg-report/README.md`

**Interfaces:**
- Consumes: content from `inicio/cv_detection_seg_report.docx` and `inicio/cv_repos_report.docx`.
- Produces: a compilable PDF and a pushed PRIVATE repo.

- [ ] **Step 1: Init repo and structure**

```bash
mkdir -p /home/andru/Projects_claude/cv-detection-seg-report/sections /home/andru/Projects_claude/cv-detection-seg-report/figures
cd /home/andru/Projects_claude/cv-detection-seg-report
git init
```

- [ ] **Step 2: Write `.gitignore`**

```
*.aux
*.log
*.out
*.bbl
*.blg
*.toc
*.synctex.gz
main.pdf
```

- [ ] **Step 3: Write `references.bib`** with the 9 arXiv references from the docx (Sapkota 2024 YOLO review, Tian 2025 YOLOv12, Jegham 2025 benchmark, Robinson 2026 RF-DETR, Cheng 2024 YOLO-World, Liu 2023 GroundingDINO, Cheng 2022 Mask2Former, Ravi 2024 SAM 2, Yakubovskiy SMP) as `@article`/`@misc` entries with `eprint` arXiv IDs.

- [ ] **Step 4: Write `main.tex`** (article class) including `sections/01..05` and `\bibliography{references}`, with title "Detection & Pixel-wise Segmentation: State of the Art and a Reproducible Benchmark", author "S. Duarte Pacheco (ORCID 0009-0002-9062-0336)".

- [ ] **Step 5: Write the five section files**, condensing the docx:
  - `01-introduction.tex` — task definitions, scope, purpose.
  - `02-detection.tex` — YOLO family evolution, DETR/RF-DETR, open-vocab; COCO comparison table.
  - `03-segmentation.tex` — Mask2Former, SAM 2, SMP.
  - `04-repositories.tex` — reference repositories table (license, stars, arXiv).
  - `05-benchmark-design.tex` — the `cv-detection-seg-benchmark` design, v0.1 scope, roadmap.

- [ ] **Step 6: Build the PDF**

```bash
cd /home/andru/Projects_claude/cv-detection-seg-report
pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Expected: `main.pdf` produced with no unresolved citations (check `main.log` for `Citation` warnings). If `pdflatex` is not installed, report it and ask the user to install TeX Live (`sudo apt install texlive-latex-recommended texlive-bibtex-extra`).

- [ ] **Step 7: Write `README.md`** with build instructions and commit everything

```bash
git add -A && git commit -m "feat: condensed LaTeX report on detection & segmentation"
```

- [ ] **Step 8: PAUSE — request the empty private repo**

Ask the user to create an **empty PRIVATE** repo named `cv-detection-seg-report` at
github.com. Do not proceed until confirmed.

- [ ] **Step 9: Add remote and push**

```bash
git branch -M main
git remote add origin git@github.com:andrudebaran7/cv-detection-seg-report.git
git push -u origin main
```

Expected: push succeeds.

---

## Self-Review

- **Spec coverage:** Repo A scaffold (T1), unified interface (T2), YOLO (T3), SAM 2 (T4), visualization (T5), Streamlit app (T6), README+push (T7), LaTeX report repo + push (T8). Lazy imports + mocked tests enforced in T3/T4. AGPL license in T1. Private Repo B in T8. All spec sections covered.
- **Placeholder scan:** Code steps contain full code; LaTeX section bodies (T8 step 5) are described by content rather than full text — acceptable, as the source material is the docx and exact prose is authored during execution. No "TODO"/"TBD" left in code.
- **Type consistency:** `Prediction(boxes, labels, scores, masks, latency_ms)` and `Box(x1,y1,x2,y2)` used identically across T2–T6; `predict()` signature consistent; `draw_prediction(image, prediction)` consistent between T5 and T6; `get_yolo/get_yolo_seg/get_sam2` consistent between T6 model_runner and pages.
