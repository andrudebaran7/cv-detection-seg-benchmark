# Project Status

_Last updated: 2026-06-27_

## Overview

Three repositories make up this project:

| Repo | Visibility | Purpose | Latest commit |
|------|-----------|---------|---------------|
| [`cv-detection-seg-benchmark`](https://github.com/andrudebaran7/cv-detection-seg-benchmark) | public | Streamlit benchmark app (code) | `8f0511d` |
| [`cv-detection-seg-report`](https://github.com/andrudebaran7/cv-detection-seg-report) | private | LaTeX technical report + PDF | `7455f41` |
| [`detection_servey`](https://github.com/andrudebaran7/detection_servey) | — | survey-template pipeline (separate) | `53f4660` |

All work is committed and pushed. Nothing is pending locally.

## Current milestone: v0.3 (shipped)

Built with TDD (18/18 unit tests green), unified `DetectionSegModel` interface, mocked
wrapper tests, and end-to-end smoke tests.

**Models (all behind `models/base.py` → `Prediction`):**
- YOLO11n — detection (`ultralytics`)
- RF-DETR-nano — detection (`rfdetr`, maps `supervision.Detections`)
- YOLO-World — open-vocabulary detection (`ultralytics`, text classes)
- YOLO11n-seg — instance segmentation (`ultralytics`)
- SAM 2 tiny — promptable segmentation (`ultralytics`)
- Mask2Former — panoptic segmentation (`transformers`, `swin-tiny-coco-panoptic`)

**Pages:** Detection, Segmentation, Open-Vocabulary, Comparison, Benchmark.

**Measured benchmark (CPU, `data/benchmark_results.csv`):**
YOLO11n 39.5 mAP / ~211 ms · RF-DETR-nano 48.4 / ~421 ms · YOLO-World 35.4 (LVIS) / ~377 ms.

## Deployment — Streamlit Community Cloud

**Live URL:** https://cv-detection-seg-benchmark-au.streamlit.app/

Deployed from `main`, entry `app/main.py`. Deploy-readiness work done:
- `requirements.txt` = CPU-only runtime (Cloud reads this); `requirements-dev.txt` adds pytest.
- `packages.txt` = `libgl1` + `libglib2.0-0t64` (system libs OpenCV/cv2 needs on the
  headless Debian-trixie image).
- `.streamlit/config.toml` (10 MB upload cap, light theme).
- `app/_pathsetup.py` puts the repo root on `sys.path` (imported first in `main.py` and
  every page), so `from app...`/`from models...` resolve under `streamlit run`.

### Verified live
- Home + all pages load (no import errors).
- **Benchmark page ran end-to-end live**: YOLO11n + YOLO-World downloaded weights, ran
  inference on the bundled image, and rendered the scatter + table (YOLO11n ~186 ms,
  YOLO-World ~190 ms on Cloud CPU).

### Bugs found by live testing (all fixed)
1. `ModuleNotFoundError` on every page — Streamlit puts `app/` (not the repo root) on
   `sys.path`. Fixed by `app/_pathsetup.py` (`0de9cad`). _Root cause: the headless boot
   check only loaded the home page, which imports nothing._
2. `ImportError: libGL.so.1` — cv2 (via ultralytics) needs system libs. Fixed by
   `packages.txt` (`984321a` → `4a448d4`).
3. `ImportError: libgthread-2.0.so.0` — next cv2 dep; on Debian trixie the package is
   `libglib2.0-0t64` (time64), not `libglib2.0-0`. Fixed (`8f0511d`).

## Open issues / known limitations

- **Free-tier RAM (~1 GB) OOM.** Loading the heaviest models together (RF-DETR 349 MB +
  Mask2Former ~190 MB + others) crashes the app ("Oh no. Error running app"). Single-model
  pages fit; the Comparison/Benchmark pages with 2 light detectors fit; 3 heavy models do
  not. Documented in the README deploy note.
- **App was rebooted** after an OOM crash during live testing; it auto-recovers ("in the
  oven") but heavy pages will OOM again on the free tier.
- **Detection live upload not yet exercised.** The Streamlit file uploader runs inside an
  iframe, so browser automation can't reach the `<input type=file>`; the upload tool also
  only accepts session-shared files. The Detection model path itself is already proven
  live (same `YoloWrapper.predict` as the Benchmark run). Remaining: a human uploads an
  image on the Detection page to confirm the upload→draw path.

## Possible next steps (not started)

- Reduce OOM risk on free tier: e.g. load one model at a time / drop cached models, or
  cap the Comparison/Benchmark pages to lighter detectors by default; or document that
  heavy combos need a paid tier.
- Finish the Detection live check (human-driven upload).
- v0.4 roadmap: RF-DETR-Seg (once its mask-prediction bug, roboflow/rf-detr#403, is
  resolved), video input, real-time metrics.
- Specs/plans for each milestone live in `docs/superpowers/specs/` and
  `docs/superpowers/plans/`.
