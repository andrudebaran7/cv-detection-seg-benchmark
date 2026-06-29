# Phase 2: Homogeneous Performance Campaign + Paper Integration

_Design spec — 2026-06-29_

## Context

Phase 1 reoriented the companion report (`cv-detection-seg-report`) from a survey into a
reproducible-benchmark + engineering study, leaving the experimental findings as
observational notes and marking a homogeneous campaign as future work with `% [Phase 2:
...]` placeholders in the Results section. See
`cv-detection-seg-report/docs/superpowers/specs/2026-06-29-paper-reorientation-design.md`.

Phase 2 turns those placeholders into real, measured content: a portable
performance-measurement harness run on CPU (local) and GPU (cloud), producing comparable
results that feed new tables and figures in the paper's Results section.

The report's central honesty caveat stays intact: **accuracy (mAP/AP/PQ) remains
published; only the performance axis (latency, memory, throughput, scaling) becomes
homogeneous and author-measured.**

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| GPU access | **B** — cloud GPU (Colab/Kaggle). Harness is device-portable; the author runs the GPU pass. |
| Measurement scope | **A** — performance axis only (latency/memory/throughput/scaling). No own accuracy. |
| Image protocol | **B** — a fixed bundled set of ~8–10 varied images committed to the repo, plus a resolution sweep. |
| Models | **C** — all 6 models, reported in separate detection and segmentation tables. |

## Repos touched

- **`cv-detection-seg-benchmark`** (code): new `benchmark/` harness package, device support
  in wrappers, plotting script, harness tests, bundled images, result CSVs, generated
  figures.
- **`cv-detection-seg-report`** (docs): Results §4 rewrite (replace placeholders), figure
  includes, separate det/seg tables, Discussion/Conclusions update, deferred minor polish.

## Architecture: the measurement harness (benchmark repo)

A standalone `benchmark/` package, separate from the Streamlit `app/`, reusing the
`models/` wrappers (the `DetectionSegModel` interface). CLI entry point:

```
python -m benchmark.run --device {cpu,cuda} --models all --out data/phase2/
```

Components:

1. **Device support in wrappers** (`models/*_wrapper.py`): add an optional `device`
   parameter to each wrapper's `__init__` (lazy-load the model onto that device).
   Default preserves current behavior (CPU/auto) so the Streamlit app is unaffected.
   This is the central code change; built with TDD using the repo's existing mocked-wrapper
   test pattern.
2. **`benchmark/measure.py`**: measurement primitives — explicit warmup, then N timed
   iterations (report mean/median/std); peak-memory capture (CPU RSS via `psutil` and the
   existing `malloc_trim` machinery; GPU via `torch.cuda.max_memory_allocated`).
3. **`benchmark/run.py`**: orchestrates the campaign — iterates (model × image × resolution
   × experiment), writes CSVs plus a **hardware manifest** (CPU model, RAM, GPU model,
   `torch`/CUDA versions, OS) for reproducibility.
4. **Colab portability**: device-agnostic; the author runs `--device cuda` in the cloud via
   the same code path; GPU CSVs are merged with the CPU CSVs by the plotting/table step.

Per-model handling (choice C): SAM 2 uses a fixed prompt (center point); Mask2Former uses
its standard panoptic call. Detection and segmentation are measured identically but reported
in separate tables.

## Experiments (map 1:1 to the Results §4 placeholders)

All 6 models, CPU (local) and GPU (Colab), over the fixed ~8–10 image set.

| # | Experiment | Protocol | Output |
|---|-----------|----------|--------|
| B1 | CPU vs GPU warm latency | resolution 640; warmup + N=50 iters; mean±std per model/device | main table (det/seg separate) + CPU-vs-GPU bar figure |
| B2 | Cold-start vs warm-start | first inference (incl. weight load + on-demand downloads) vs steady-state median | cold/warm table per model |
| B3 | Throughput | sustained images/sec over the full set, steady-state | column in main table (or own table) |
| B4 | Image-size scaling | one base image resized to {320, 640, 960, 1280}; latency + peak memory at each | line figure latency-vs-resolution + memory-vs-resolution, per model |
| B5 | Memory | peak RSS (CPU) and `max_memory_allocated` (GPU) per model | extends the existing memory table with a GPU column |

Cross-cutting homogeneity rules (the "homogeneous protocol" the reviews asked for):
- Same image set, same preprocessing, same implementation, same iteration count on both devices.
- Explicit warmup before timing (discards cold-start except in B2).
- Report mean ± std, never a single value — this is what raises the numbers from
  "indicative" to "systematic."
- The hardware manifest accompanies every CSV.

## Data outputs and reproducibility (benchmark repo)

- CSVs under `data/phase2/` — one per experiment, long format with columns:
  `device, model, task, image, resolution, experiment, metric, value, n_iters, mean, std,
  measured_at`.
- A `data/phase2/manifest_{cpu,cuda}.json` capturing the host hardware/software for each
  device run.
- Determinism: fixed warmup + iteration counts; the run is re-runnable and overwrites its
  own CSV for a given device.

## Figures

Three kinds, kept reproducible where possible:

1. **Data plots** (`benchmark/plot.py`): generated from the CSVs (CPU-vs-GPU bars,
   latency-vs-resolution, memory-vs-resolution). Scripted and reproducible; output to
   `cv-detection-seg-report/figures/` as PDF/PNG.
2. **Architecture & flow diagrams**: the `DetectionSegModel → Prediction → app/benchmark`
   architecture and the system flow. Authored as TikZ in the report (preferred for
   reproducibility) or as a committed vector file in `figures/`.
3. **App screenshots**: captured from the five live app pages (home, detection,
   segmentation, comparison, benchmark) — a manual, human-in-the-loop step.

## Paper integration (report repo)

- Replace the `% [Phase 2: ...]` placeholders in `sections/04-results.tex` with real
  subsections, the separate detection/segmentation tables, and figure includes.
- Restore an accurate unit-test count: `pytest --collect-only` reports **22 tests** (the
  paper's original "(22 tests)" was correct; `STATUS.md`'s "18/18" is stale). State 22 in
  the paper and fix `STATUS.md` in the benchmark repo.
- Update the Discussion and Conclusions: the homogeneous campaign is now **done**, not
  "planned" — soften the published-vs-measured caveat accordingly while keeping the
  accuracy-is-borrowed honesty.
- Apply the deferred minor polish from the Phase 1 final review:
  - abstract: add "(detection and segmentation)" after YOLO11 to pre-empt the "six
    models / five names" miscount;
  - reword the printed "Phase 2" jargon (method/discussion) to a reader-facing phrase
    ("a future homogeneous campaign" → now "this campaign");
  - remove the orphaned bib entry `yakubovskiy2019smp` (SMP) from `references.bib`.
- Final full bibliography build, zero unresolved references.

## Implementation decomposition (two plans, with a human handoff)

This spec is implemented as **two sequential plans** because a cloud GPU run by the author
sits between them:

- **Plan 2a — harness, experiments, CPU data, data plots** (benchmark repo). Deliverables:
  the `benchmark/` package, device support, harness tests, bundled images, CPU CSVs +
  manifest, and the plotting script. Ends with the author running the same harness on a
  cloud GPU to produce the GPU CSVs + manifest.
- **Plan 2b — paper integration** (report repo). Runs once both CPU and GPU CSVs exist:
  generate final figures, write the Results subsections/tables, apply polish, update
  Discussion/Conclusions, final build.

## Out of scope — Future Work (explicitly deferred)

These are recorded here and belong in the paper's Future Work, not in Phase 2:

- **Own accuracy/mAP evaluation** on a labeled held-out COCO/LVIS split (would make the
  accuracy axis homogeneous too) — deferred per decision A.
- **Export/optimization backends**: ONNX, OpenVINO, TensorRT, and quantization (int8/fp16)
  performance comparisons.
- **Batched and multi-GPU** inference; concurrency/load testing under multiple users.
- **Additional datasets**: ADE20K, Cityscapes, RF100-VL — broader evaluation surfaces.
- **More models**: GroundingDINO, Florence-2, EfficientSAM, FastSAM, MobileSAM, YOLO26,
  RF-DETR-Seg (once roboflow/rf-detr#403 resolves).
- **Real-time/video input** and live streaming metrics (the v0.4 roadmap item).

## Success criteria

- The harness runs end-to-end on CPU locally and on a cloud GPU via the same command,
  producing the five experiments' CSVs + a hardware manifest per device.
- All harness code is covered by mocked tests consistent with the repo's existing suite,
  and the full suite stays green.
- The paper's Results §4 contains real measured tables (det/seg separate) and figures with
  no remaining `[Phase 2: ...]` placeholders; the test-count claim is accurate (22);
  Discussion/Conclusions reflect a completed campaign; final build has zero unresolved refs.
