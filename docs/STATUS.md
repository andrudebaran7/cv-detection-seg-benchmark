# Project Status

_Last updated: 2026-06-30_

## Overview

Three repositories make up this project:

| Repo | Visibility | Purpose | Latest commit |
|------|-----------|---------|---------------|
| [`cv-detection-seg-benchmark`](https://github.com/andrudebaran7/cv-detection-seg-benchmark) | public | Streamlit benchmark app + measurement harness | `55b1dbb` |
| [`cv-detection-seg-report`](https://github.com/andrudebaran7/cv-detection-seg-report) | private | LaTeX technical report (IEEE 2-column) + PDF | `56bd574` |
| [`detection_servey`](https://github.com/andrudebaran7/detection_servey) | — | survey-template pipeline (separate) | `53f4660` |

All work is committed and pushed. Nothing is pending locally.

## Current milestone: v0.3 (shipped)

Built with TDD (55/55 unit tests green on `main`), unified `DetectionSegModel` interface,
mocked wrapper tests, and end-to-end smoke tests. A Phase 2 measurement harness
(`benchmark/`) was added on top, giving device-portable CPU/GPU performance measurement
(latency, memory, throughput, image-size scaling) with results written to `data/phase2/`.

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

## Memory / OOM mitigation (2026-06-27)

`app/components/model_runner.py` now uses a **single-slot loader**: at most one model is
held in memory at a time. Requesting a different model evicts the previous one
(`load_model(key)` / `_load_into_slot`). Crucially, eviction calls `gc.collect()` **and
`malloc_trim(0)`** — dropping a reference + gc alone does NOT lower RSS (CPU torch and
glibc retain freed memory in their own pools). The Comparison and Benchmark pages now
load-run-release detectors **one at a time** via `load_model`.

Measured on the dev machine (CPU), loading the three detectors in sequence:

| step | RSS (resident) | after evict + `malloc_trim` |
|------|----------------|------------------------------|
| YOLO11n | ~439 MB | ~410 MB (framework floor) |
| RF-DETR-nano | ~986 MB | ~523 MB |
| YOLO-World | ~1337 MB | ~557 MB |

So `malloc_trim` reclaims each model's weights after use — this removes the **additive
accumulation** that caused the live OOM (models piling up across page visits).

**Live re-test (2026-06-27):** with the single-slot + `malloc_trim` mitigation deployed,
the Benchmark page ran **all three heavy detectors** (YOLO11n + RF-DETR-nano + YOLO-World)
one at a time **without OOM** — the same combo that crashed before. Measured on Cloud CPU:
YOLO11n 39.5 mAP / 188 ms, YOLO-World 35.4 / ~330 ms, RF-DETR-nano 48.4 / 686 ms. The
upload cap was also lowered to 5 MB.

**Remaining caveat:** the framework floor still grows as torch / ultralytics / rfdetr /
transformers / CLIP get imported, and load transients are large, so headroom on the free
tier is thin — concurrent users or repeated heavy runs could still hit the limit. Single-
model pages are comfortable; the heavy Comparison/Benchmark works but with little margin
(a paid tier removes the risk).
- **Detection live upload not yet exercised.** The Streamlit file uploader runs inside an
  iframe, so browser automation can't reach the `<input type=file>`; the upload tool also
  only accepts session-shared files. The Detection model path itself is already proven
  live (same `YoloWrapper.predict` as the Benchmark run). Remaining: a human uploads an
  image on the Detection page to confirm the upload→draw path.

## Phase 2 + companion paper — final state (2026-06-30)

Both repos are complete on `main` and pushed to GitHub.

**Phase 2 — homogeneous performance campaign (`benchmark/` package, separate from `app/`):**
a portable, device-agnostic measurement harness measuring all six models on CPU (local) and
GPU (cloud Tesla T4 via Colab) under one protocol. Wrappers gained an optional `device`
parameter (default `None` preserves prior behaviour). Experiments: warm latency (CPU vs GPU),
cold-start vs warm-start, throughput, image-size scaling, and peak memory (host RSS on CPU,
CUDA VRAM on GPU). Results in `data/phase2/results_{cpu,cuda}.csv` (+ hardware manifests);
GPU speedups range ~4×–45× over CPU. Run it with `python -m benchmark.run --device {cpu,cuda}`;
the Colab GPU pass is documented in `docs/COLAB_RUNBOOK.md` / `docs/colab_gpu_campaign.ipynb`.
Two GPU-only device bugs (YOLO-World `set_classes` before `.to(device)`; Mask2Former cuda→numpy)
were found on Colab and fixed. Suite: 55/55 green.

**Companion paper (`cv-detection-seg-report`):** reoriented from a survey into a
reproducible-benchmark + engineering study, then converted to **IEEE two-column format**
(`IEEEtran` conference class with a `twocolumn`-article fallback; build via `make`). It carries
the real CPU+GPU tables (separate detection/segmentation), the gap and scaling figures
(generated from the CSVs by `benchmark/build_report_assets.py`), and a TikZ system-architecture
diagram. Author: Sergio Duarte (Independent Researcher). Builds clean: 8 pages, zero unresolved
references, zero overfull boxes.

**Optional remaining work (deferred, not blocking):** app screenshots (dropped — Streamlit
upload iframe friction); averaging the campaign over all 8 bundled images (currently the sweep
uses one base image); own accuracy/mAP evaluation; ONNX/OpenVINO/quantization; arXiv/workshop
submission.

## Audit + corrections (2026-07-01)

An independent audit (`audicion_revision/` in both repos) cross-checked the paper's numbers
against the CSVs and the harness. It found real defects, all since fixed and independently
re-verified:
- **RF-DETR label bug (real):** the wrapper indexed a 0..79 list but RF-DETR emits COCO
  91-scheme category ids (1..90) --- it mislabelled person/bus as bicycle/train. Fixed with a
  proper id map (`models/rfdetr_wrapper.py`) + regression test.
- **Memory measurement:** now measured per model in an isolated subprocess with a true peak
  (`benchmark/mem_probe.py`), and the campaign measures **all memory first** (two-phase
  `run.main`) so a heavy parent process cannot inflate the child's `ru_maxrss`. Earlier
  in-process readings double-counted retained pages.
- **Paper rigor:** dropped a duplicate/contradictory latency table; declared the ~15 GB
  measurement host (distinct from the ~1 GB Streamlit tier); cited every published claim in
  the detection table; noted CPU/GPU memory are not the same quantity; precise speedup range
  (CPU is ~3--33x slower than our own T4).
- **GPU memory refreshed (2026-07-01):** the Colab campaign was re-run with the corrected
  two-phase harness, so `data/phase2/results_cuda.csv` now carries isolated per-model VRAM
  (`measured_at 2026-07-01`); the report's memory table/figure were regenerated to match.

## Possible next steps (not started)

- Reduce OOM risk on free tier: e.g. load one model at a time / drop cached models, or
  cap the Comparison/Benchmark pages to lighter detectors by default; or document that
  heavy combos need a paid tier.
- Finish the Detection live check (human-driven upload).
- v0.4 roadmap: RF-DETR-Seg (once its mask-prediction bug, roboflow/rf-detr#403, is
  resolved), video input, real-time metrics.
- Specs/plans for each milestone live in `docs/superpowers/specs/` and
  `docs/superpowers/plans/`.
