# cv-detection-seg-benchmark — v0.2 Design

**Date:** 2026-06-26
**Author:** S. Duarte Pacheco (with Claude)
**Status:** Approved for implementation planning
**Builds on:** v0.1 (see `2026-06-26-cv-detection-seg-benchmark-v0.1-design.md`)

## Context

v0.1 shipped the unified interface (`models/base.py`), YOLO and SAM 2 wrappers, and a
minimal Streamlit app (home, detection, segmentation pages), all tested with mocked
models. v0.2 extends the benchmark with two new detectors and a side-by-side
comparison page, following the same patterns.

## Goals (v0.2)

1. Add **RF-DETR** (detection only) as a wrapper implementing `DetectionSegModel`.
2. Add **YOLO-World** (open-vocabulary detection) as a wrapper implementing
   `DetectionSegModel`, driven by user-supplied text classes.
3. Add a **comparison page**: run 2–3 detectors on one image and show results
   side-by-side with a metrics table (object count, latency).
4. Let the existing Detection page choose between YOLO11n and RF-DETR.
5. Add an **Open-Vocabulary page** for YOLO-World with predefined COCO classes plus
   free-text classes.

## Non-Goals (deferred)

- **RF-DETR-Seg** → v0.3. It currently ships only under the `RFDETRSegPreview` class
  (preview state) with a known open mask-prediction bug
  ([roboflow/rf-detr#403](https://github.com/roboflow/rf-detr/issues/403)). Deferring
  keeps v0.2 stable.
- Mask2Former, benchmark page (latency vs mAP), video input → v0.3+.

## Global Constraints (unchanged from v0.1, plus v0.2 additions)

- Python >= 3.9. Use the existing project venv `.venv`.
- Repo license remains **AGPL-3.0**. New deps `rfdetr` and `supervision` are Apache-2.0
  (compatible).
- Default weights remain **nano/tiny**: RF-DETR uses `RFDETRNano`; YOLO-World uses
  `yolov8s-world.pt` (small). Streamlit Community Cloud ~1 GB RAM budget.
- All wrappers MUST **lazy-import** their heavy backend (import inside methods).
- Tests MUST **mock** the underlying model — never download weights in tests.
- `Prediction` (from `models/base.py`) remains the only data contract; `predict()`
  accepts `**kwargs`, so YOLO-World's `classes` argument fits without changing the ABC.
- Comparison page loads models **on demand**, caps simultaneous models at **3**, and is
  **detection-focused** (comparable boxes).

## New Components

### `models/rfdetr_wrapper.py`

- `class RfDetrWrapper(DetectionSegModel)`, `__init__(self, model_name: str = "nano")`.
- `_load()` lazy-imports `rfdetr` and instantiates the matching class (`RFDETRNano`),
  cached on `self._model`.
- `predict(image, threshold: float = 0.5, **kwargs) -> Prediction`: calls
  `model.predict(image, threshold=threshold)`, which returns a `supervision.Detections`
  object. Maps `det.xyxy` → `Box`, `det.confidence` → scores, `det.class_id` →
  labels via the COCO class names (`rfdetr.util.coco_classes` / bundled list);
  `masks=None`; `latency_ms` measured with `time.perf_counter`.
- Detection only in v0.2.

### `models/yoloworld_wrapper.py`

- `class YoloWorldWrapper(DetectionSegModel)`, `__init__(self, weights: str = "yolov8s-world.pt")`.
- `_load()` lazy-imports `ultralytics.YOLOWorld`, cached on `self._model`.
- `predict(image, classes: list[str] | None = None, conf: float = 0.25, **kwargs) -> Prediction`:
  raises `ValueError` if `classes` is empty/None; calls `model.set_classes(classes)`
  then runs inference; maps output to `Prediction` like `YoloWrapper` (boxes, labels
  from the supplied classes, scores, latency).

## UI Changes

- **`app/pages/1_Detection.py`** — add a model selector (`YOLO11n`, `RF-DETR-nano`);
  dispatch to the chosen wrapper via `model_runner`.
- **`app/pages/3_OpenVocab.py`** (new) — `st.multiselect` of common COCO classes +
  an `st.text_input` for extra comma-separated classes; runs `YoloWorldWrapper`.
- **`app/pages/4_Comparison.py`** (new) — upload one image, `st.multiselect` of 2–3
  detectors, run each, render images side-by-side (`st.columns`) and a metrics table
  (model, #objects, latency ms).
- **`app/components/model_runner.py`** — add `@st.cache_resource` loaders
  `get_rfdetr()` and `get_yoloworld()`.
- **`app/components/comparison.py`** (new) — pure helper
  `run_comparison(image, models: dict[str, DetectionSegModel], **per_model_kwargs)
  -> list[ComparisonResult]` where `ComparisonResult` carries `name`, `prediction`,
  and the rendered image. Keeps the page thin and the logic unit-testable.

Note: the Segmentation page (`2_Segmentation.py`) is unchanged in v0.2. Page numbering
keeps SAM 2 segmentation at position 2; Open-Vocab is 3 and Comparison is 4.

## Dependencies

Add to both `requirements.txt` and `requirements-light.txt`:

```
rfdetr>=1.8
supervision>=0.20
```

## Testing

- `tests/test_rfdetr_wrapper.py` — mock `rfdetr.RFDETRNano` and a `supervision.Detections`
  -like object; assert mapping to `Prediction` (boxes/labels/scores, masks None,
  latency populated) and lazy import.
- `tests/test_yoloworld_wrapper.py` — mock `ultralytics.YOLOWorld`; assert `set_classes`
  is called with the supplied classes, mapping to `Prediction`, and that empty classes
  raise `ValueError`.
- `tests/test_comparison.py` — `run_comparison` with two mocked `DetectionSegModel`s
  returns one `ComparisonResult` per model, preserving order, each with a `Prediction`
  and a rendered image (reuse `draw_prediction`).

## Error Handling

- Wrappers surface a clear error if their backend/weights are unavailable; the Streamlit
  pages catch and show a friendly message rather than a traceback.
- YOLO-World page disables the run button until at least one class is selected/entered.
- Comparison page enforces the 2–3 model selection bound in the UI.

## Risks

- `rfdetr.predict()` returns a `supervision.Detections` object whose attribute shape
  must be confirmed during implementation (`.xyxy`, `.confidence`, `.class_id`). The
  wrapper's mapping test mocks this contract; the real shape is verified in the
  end-to-end smoke test.
- COCO class-name source for RF-DETR: confirm the exact import path
  (`rfdetr.util.coco_classes` or equivalent) during implementation; fall back to a
  bundled 80-class list if the import path differs.

## Roadmap (after v0.2)

- **v0.3** — RF-DETR-Seg (once out of preview), Mask2Former; benchmark page
  (latency vs mAP).
- **v0.4** — video input; real-time metrics.
- **v1.0** — full docs, demo GIF, Zenodo release.
