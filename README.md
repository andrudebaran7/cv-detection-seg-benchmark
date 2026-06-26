# cv-detection-seg-benchmark

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Milestone](https://img.shields.io/badge/milestone-v0.3-orange)

Interactive comparison of object **detection** and image **segmentation** models
through a single unified interface, with a Streamlit demo. Built to fit
Streamlit Community Cloud (nano/tiny models only).

## Models

| Model | Task | Weights | Source |
|-------|------|---------|--------|
| YOLO11n | Detection | `yolo11n.pt` | [ultralytics](https://github.com/ultralytics/ultralytics) |
| RF-DETR-nano | Detection | `RFDETRNano` | [rf-detr](https://github.com/roboflow/rf-detr) |
| YOLO-World | Open-vocabulary detection | `yolov8s-world.pt` | [ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLO11n-seg | Instance segmentation | `yolo11n-seg.pt` | [ultralytics](https://github.com/ultralytics/ultralytics) |
| SAM 2 tiny | Promptable segmentation | `sam2.1_t.pt` | [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) |
| Mask2Former | Panoptic segmentation | `mask2former-swin-tiny-coco-panoptic` | [transformers](https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic) |

All models implement the same `models/base.py` interface (`predict() -> Prediction`),
so the app is independent of the concrete model. A later milestone adds RF-DETR-Seg
(see roadmap).

## Pages

- **Detection** — YOLO11n or RF-DETR-nano on an uploaded image.
- **Segmentation** — YOLO11n-seg (automatic), SAM 2 tiny (point-prompted), and
  Mask2Former (panoptic).
- **Open-Vocabulary** — YOLO-World with predefined COCO classes plus free-text classes.
- **Comparison** — run 2–3 detectors side-by-side with a metrics table.
- **Benchmark** — measured on-device latency vs published COCO mAP (scatter + table).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # GPU/CUDA torch
.venv/bin/streamlit run app/main.py
```

Weights download automatically on first use. Two sample images live in
`data/sample_images/`.

## Tests

Tests mock the underlying models (no weight downloads), so the suite is fast:

```bash
.venv/bin/python -m pytest -v
```

## Deploy to Streamlit Community Cloud

Use `requirements-light.txt` (CPU-only torch) to stay within the ~1 GB RAM limit:

```bash
.venv/bin/pip install -r requirements-light.txt
```

Point the Streamlit Cloud app at `app/main.py` and set the requirements file to
`requirements-light.txt`.

## Architecture

```
models/base.py        # Prediction dataclass + DetectionSegModel ABC (the contract)
models/yolo_wrapper.py
models/sam2_wrapper.py
models/rfdetr_wrapper.py
models/yoloworld_wrapper.py
models/mask2former_wrapper.py
app/main.py           # Streamlit home
app/pages/            # 1_Detection, 2_Segmentation, 3_OpenVocab, 4_Comparison, 5_Benchmark
app/components/       # model_runner (caching), visualization, comparison, benchmark
```

## Roadmap

- **v0.1** — base structure, YOLO + SAM 2, minimal Streamlit app. ✅
- **v0.2** — RF-DETR (detection), YOLO-World; side-by-side comparison page. ✅
- **v0.3** (this release) — Mask2Former (panoptic); benchmark page (latency vs mAP). ✅
- **v0.4** — RF-DETR-Seg (once masks return reliably); video input; real-time metrics.
- **v1.0** — full docs, demo GIF, Zenodo release.

## Companion report

A technical report on the state of the art in detection & segmentation accompanies
this repository (private): `cv-detection-seg-report`.

## License

AGPL-3.0. This project depends on `ultralytics` (AGPL-3.0); commercial use of those
models may require an Ultralytics Enterprise license.
```
