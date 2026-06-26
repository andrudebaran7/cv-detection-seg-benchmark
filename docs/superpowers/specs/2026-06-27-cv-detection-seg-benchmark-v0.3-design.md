# cv-detection-seg-benchmark — v0.3 Design

**Date:** 2026-06-27
**Author:** S. Duarte Pacheco (with Claude)
**Status:** Approved for implementation planning
**Builds on:** v0.2 (RF-DETR detection, YOLO-World, comparison page)

## Context

v0.1 shipped the unified `DetectionSegModel` interface with YOLO and SAM 2; v0.2 added
RF-DETR detection, YOLO-World open-vocabulary detection, and a comparison page. v0.3
adds Mask2Former (universal segmentation) and a benchmark page, following the same
patterns.

## Goals (v0.3)

1. Add **Mask2Former** (panoptic segmentation) as a wrapper implementing
   `DetectionSegModel`, via HuggingFace `transformers`.
2. Add Mask2Former as a third option on the Segmentation page.
3. Add a **benchmark page** that measures each detector's latency on-device and plots
   it against published COCO mAP.

## Non-Goals (deferred)

- **RF-DETR-Seg** → still deferred. The mask-prediction bug
  ([roboflow/rf-detr#403](https://github.com/roboflow/rf-detr/issues/403)) remains
  unresolved and the API is still `RFDETRSegPreview`; revisit when masks return
  reliably.
- Video input, real-time metrics → v0.4+.

## Global Constraints (carried from v0.2, plus v0.3 additions)

- Python >= 3.9. Use the existing project venv `.venv`.
- Repo license remains **AGPL-3.0**. New dep `transformers` is Apache-2.0 (compatible).
- Mask2Former uses **`facebook/mask2former-swin-tiny-coco-panoptic`** (~190 MB) to fit
  the Streamlit Community Cloud ~1 GB RAM budget. (HuggingFace hosts no ResNet-50
  Mask2Former; Swin-tiny is the smallest panoptic checkpoint. Swin-L is too large.)
- All wrappers MUST **lazy-import** their backend (import inside methods).
- Tests MUST **mock** the backend — never download weights in tests.
- `Prediction` (from `models/base.py`) remains the only data contract.
- Benchmark page is **detection-focused** (mAP is comparable across detectors);
  Mask2Former (panoptic PQ, not mAP) is not placed on the mAP scatter.
- No new plotting dependency: use Streamlit's native `st.scatter_chart`.
- Commit message bodies end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`;
  commit with `git -c commit.gpgsign=false`.

## New Components

### `models/mask2former_wrapper.py`

- `class Mask2FormerWrapper(DetectionSegModel)`,
  `__init__(self, model_id: str = "facebook/mask2former-swin-tiny-coco-panoptic")`.
- `_load()` lazy-imports `transformers.AutoImageProcessor` and
  `transformers.Mask2FormerForUniversalSegmentation`, caches a `(processor, model)`
  tuple on `self._model`.
- `predict(image, **kwargs) -> Prediction`:
  - processes the image, runs the model, calls
    `processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]`
    which yields `{"segmentation": HxW id map, "segments_info": [{"id","label_id","score"}, ...]}`.
  - For each segment: build a boolean mask `(segmentation == id)`, label via
    `model.config.id2label[label_id]`, score from `segments_info`.
  - Returns `Prediction(boxes=[], labels=[...], scores=[...], masks=[...], latency_ms=...)`.

### `app/components/benchmark.py`

- `PUBLISHED_MAP: dict[str, dict]` — per detector: `{"map": float, "note": str}`. Values
  from the report's COCO table; YOLO-World noted as LVIS zero-shot.
- `def measure_latency(model: DetectionSegModel, image, **kwargs) -> float` — runs
  `model.predict(image, **kwargs)` and returns `prediction.latency_ms`.
- `def build_benchmark_rows(measured: dict[str, float]) -> list[dict]` — joins measured
  latency with `PUBLISHED_MAP`, returning rows `{"model", "map", "latency_ms", "note"}`
  for models present in both, preserving `measured` order.

## UI Changes

- **`app/pages/2_Segmentation.py`** — add `Mask2Former (panoptic)` to the model radio;
  dispatch to `get_mask2former()`.
- **`app/pages/5_Benchmark.py`** (new) — multiselect of detectors (YOLO11n, RF-DETR-nano,
  YOLO-World), run each on a bundled sample image to measure latency, then render a
  `st.scatter_chart` of mAP (y) vs latency (x) and a `st.table` of the rows. YOLO-World
  requires a small fixed class list (e.g. the common COCO subset) for its run.
- **`app/components/model_runner.py`** — add `@st.cache_resource get_mask2former()`.

## Dependencies

Add to both `requirements.txt` and `requirements-light.txt`:

```
transformers>=4.40
```

## Testing

- `tests/test_mask2former_wrapper.py` — mock `transformers.AutoImageProcessor` and
  `transformers.Mask2FormerForUniversalSegmentation`; the processor's
  `post_process_panoptic_segmentation` returns a crafted segmentation map +
  `segments_info`. Assert mapping to `Prediction` (one mask per segment, labels via
  `id2label`, `boxes == []`, latency populated) and lazy import.
- `tests/test_benchmark.py` — `measure_latency` with a mocked model returns the
  prediction's `latency_ms`; `build_benchmark_rows` joins measured latency with
  `PUBLISHED_MAP`, preserves order, and includes only models present in both.

## Error Handling

- Mask2Former wrapper surfaces a clear error if `transformers`/weights are unavailable;
  the Segmentation page catches it and shows a friendly message.
- Benchmark page warns if fewer than two detectors are selected.

## Risks

- `post_process_panoptic_segmentation` output shape (`segmentation` tensor +
  `segments_info` keys `id`/`label_id`/`score`) is verified against the installed
  `transformers` in the end-to-end smoke test; the unit test mocks this contract.
- `transformers` + its torch dependency add install weight; CPU wheels are used via the
  existing `requirements-light.txt` torch index.

## Roadmap (after v0.3)

- **v0.4** — RF-DETR-Seg (once masks return reliably); video input; real-time metrics.
- **v1.0** — full docs, demo GIF, Zenodo release.
