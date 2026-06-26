# cv-detection-seg-benchmark — v0.1 Design

**Date:** 2026-06-26
**Author:** S. Duarte Pacheco (with Claude)
**Status:** Approved for implementation planning

## Context

Two Word reports (in `inicio/`) define a planned project, `cv-detection-seg-benchmark`:
an interactive comparison tool for object detection and image segmentation models
(YOLO family, RT-DETR/RF-DETR, SAM 2, Mask2Former, etc.) with a Streamlit demo.
This spec covers the **first milestone (v0.1)** plus a companion **LaTeX report**.

The user wants the full benchmark eventually, decomposed by the roadmap in the docs
(v0.1 → v1.0). v0.1 establishes a solid, deployable base; later models are added in
subsequent milestones/sessions.

## Goals (v0.1)

1. A working Streamlit app that runs object **detection** and **segmentation** on an
   uploaded image, using lightweight models suitable for Streamlit Community Cloud.
2. A **unified model interface** (`models/base.py`) so the app is independent of the
   concrete model — the architectural "next step" the source reports call out.
3. A clean, focused **LaTeX report** condensing the two docx reports.
4. Delivered as two GitHub repos under `andrudebaran7`: Repo A public, Repo B private.

## Non-Goals (deferred to v0.2+)

- RF-DETR, YOLO-World, GroundingDINO, Mask2Former wrappers.
- Side-by-side multi-model comparison page; COCO eval; latency profiling page.
- Video input; real-time metrics.

## Deliverables — Two Repositories

### Repo A — `cv-detection-seg-benchmark` (public, AGPL-3.0)

Rationale for AGPL-3.0: `ultralytics` and `yolov12` are AGPL-3.0 (strong copyleft);
a public repo depending on them is cleanest licensed AGPL-3.0 as well.

```
cv-detection-seg-benchmark/
├── app/
│   ├── main.py                 # Streamlit entry: intro + how-to + sample images
│   ├── pages/
│   │   ├── 1_Detection.py      # upload image → YOLO det → boxes + labels
│   │   └── 2_Segmentation.py   # YOLO-seg (auto) + SAM 2 (point/box prompt)
│   └── components/
│       ├── model_runner.py     # @st.cache_resource model loading
│       └── visualization.py    # draw boxes, masks, labels on image
├── models/
│   ├── base.py                 # DetectionSegModel ABC + Prediction dataclass
│   ├── yolo_wrapper.py         # ultralytics YOLO (det + seg)
│   └── sam2_wrapper.py         # SAM 2 promptable via ultralytics
├── data/sample_images/         # 2–3 royalty-free example images
├── tests/
│   ├── test_base.py            # Prediction dataclass behavior
│   └── test_wrappers.py        # wrappers with MOCKED models (no weight download)
├── requirements.txt            # full: torch, ultralytics, streamlit, numpy, pillow
├── requirements-light.txt      # Streamlit Cloud: CPU torch, nano weights only
├── .gitignore
├── LICENSE                     # AGPL-3.0
└── README.md                   # badges, quickstart, model table, deploy notes
```

### Repo B — `cv-detection-seg-report` (private)

```
cv-detection-seg-report/
├── main.tex                    # paper (article class)
├── references.bib              # arXiv refs from the docx (YOLO, DETR, SAM 2, ...)
├── sections/
│   ├── 01-introduction.tex
│   ├── 02-detection.tex        # YOLO family, DETR/RF-DETR, open-vocab
│   ├── 03-segmentation.tex     # Mask2Former, SAM 2, SMP
│   ├── 04-repositories.tex     # reference repos table
│   └── 05-benchmark-design.tex # the benchmark + roadmap
├── figures/
├── .gitignore
└── README.md                   # build instructions (pdflatex + bibtex)
```

Report content: **condensed and reorganized** from the two docx (state of the art,
COCO comparison table, repository table, arXiv references), focused and aligned with
what v0.1 actually ships — not a verbatim copy.

## Architecture — Unified Interface

`models/base.py`:

- `Prediction` dataclass: `boxes: list[Box]`, `masks: list[Mask] | None`,
  `labels: list[str]`, `scores: list[float]`, `latency_ms: float`.
- `DetectionSegModel` ABC: `predict(image, **kwargs) -> Prediction`. Concrete wrappers
  (`YoloWrapper`, `Sam2Wrapper`) implement it. The Streamlit app only ever calls
  `predict()` and renders the returned `Prediction`, never touching model internals.

Data flow: `image (PIL/np) → wrapper.predict() → Prediction → visualization.draw()`.

## Models in v0.1

- **YOLO** (`yolo_wrapper.py`): `yolo11n.pt` (detection), `yolo11n-seg.pt`
  (instance segmentation), via `ultralytics`. Nano weights (~5–90 MB).
- **SAM 2** (`sam2_wrapper.py`): `sam2.1_t.pt` (tiny, point/box promptable), via the
  `ultralytics` SAM integration.

## Testing & Heavy Dependencies

`ultralytics` pulls in `torch` (large). To keep tests fast and weight-download-free:

- Wrappers use **lazy imports** (import `ultralytics` inside methods, not at module top).
- Tests **mock** the underlying model object and assert that the wrapper maps raw output
  to `Prediction` correctly (box/label/score/mask shapes, latency populated).
- Real weights download only when the app actually runs.

## Error Handling

- Wrappers raise a clear error if `ultralytics`/weights are unavailable, surfaced in the
  Streamlit UI as a friendly message (not a traceback).
- App validates uploaded file type/size before inference.

## GitHub Delivery

SSH (key present as `andrudebaran7`) allows **push** but not repo **creation**. Plan:
the **user creates two empty repos** on github.com (Repo A public, Repo B private; no
README/license/gitignore),
then Claude adds the remotes and pushes. Both built and committed locally first.

## Roadmap (beyond v0.1)

- **v0.2** — RF-DETR + YOLO-World; side-by-side comparison page.
- **v0.3** — Mask2Former (HF) ; benchmark page (latency vs mAP).
- **v0.4** — video input; real-time metrics.
- **v1.0** — full README/badges/GIF; Zenodo release.
