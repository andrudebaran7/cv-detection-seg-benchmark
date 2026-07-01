# Phase 2c: Latency Distribution over Images + 1 GB Deployment Feasibility

_Design spec — 2026-07-01_

## Context

An IEEE/CVPR-style review (`Revision_IEEE/`) rated the companion paper a "Weak Accept" and
named two technical weaknesses that the paper's own *Threats to validity* already concedes:
(1) latency is measured on essentially a **single image per resolution** — no distribution, no
percentiles, no confidence intervals; and (2) the "constrained ~1 GB" narrative is not measured
where the app actually deploys (the audit's P1). Phase 2c addresses (1) directly and adds an
honest, cheap treatment of the 1 GB constraint.

This extends the existing measurement harness (`benchmark/`) and feeds new tables/figures into
the paper. It does **not** touch the algorithmic side; accuracy stays published.

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Sampling | **~100 fixed real COCO images** (the `coco128` subset — one pinned zip), one timed inference per image after a global warmup → the latency distribution is over image **content** (~100 samples per model×device). _Deviation from the original "100 val2017 ids": coco128 is one pinned, permanent zip and far more reproducible than a fragile 100-id list; since no accuracy is measured — only latency over varied content — the specific split is immaterial._ |
| Devices | **CPU (desktop) + GPU (cloud T4)**; the GPU pass is a Colab handoff |
| Resolution | Fixed at **640px** for the distribution; the existing resolution sweep (single base image) is unchanged |
| Statistics | mean, std, **P50/P90/P99** |
| 1 GB tier | A **feasibility table** derived from the isolated per-model `peak_rss` (fits if `< 1024 MB`), not a live campaign — Streamlit Cloud can't host a scriptable harness and three models exceed 1 GB |
| Presentation | A **dedicated distribution table** (mean/std/P50/P90/P99 per model×device) **+ a box-plot figure**; the existing perf tables stay |

## Repos touched

- **`cv-detection-seg-benchmark`** (code): ~100-image fetch (coco128), a percentile primitive, a
  distribution campaign mode, its CSVs, and the generators (distribution table, box-plot, 1 GB table).
- **`cv-detection-seg-report`** (docs): a new Results subsection + the box-plot figure + the two
  new tables; threats/abstract/method updates.

## Part A — Measurement (benchmark repo)

**1. ~100-image set.** Add a `fetch_dist_images(limit=100)` to `benchmark/fetch_images.py` that
downloads the pinned **`coco128`** zip (`https://ultralytics.com/assets/coco128.zip`, 128 real COCO
images) and extracts the jpgs into `data/dist_images/`. The `.jpg` files are fetched on demand and
git-ignored (repo stays small; reproducibility comes from the pinned zip URL). The existing 8-image
val2017 set (`IMAGE_IDS`) is untouched.

**2. Percentile primitive.** Add to `benchmark/measure.py` a function
`latency_stats(samples_ms) -> {n, mean_ms, std_ms, p50_ms, p90_ms, p99_ms}` using
`statistics.quantiles(..., n=100)` (or an equivalent). Pure, unit-tested on a known list.

**3. Distribution campaign (new, separate from the per-resolution run).** A new
`benchmark/run_dist.py` (CLI `python -m benchmark.run_dist --device {cpu,cuda}
--out data/phase2`): for each model, load once, resize every one of the 100 images to 640,
run a global warmup, then **time one inference per image** → 100 samples. It writes the **raw
per-image samples** to `data/phase2/results_dist_{device}.csv`
(columns: `device, model, task, image_id, latency_ms, resolution, measured_at`) plus a
hardware manifest `manifest_dist_{device}.json`. The stats (mean/std/P50/P90/P99) are computed
downstream from the raw samples so the box-plot and the table share one source. Per-model
try/except resilience mirrors `run.py` (a failing model is skipped, others still write).
Memory is not measured here, so the two-phase memory isolation of `run.py` does not apply;
models are loaded in-process for timing, which does not affect latency.

**4. 1 GB feasibility (derived, no new measurement).** A generator reads the isolated
`peak_rss` at 640 from the existing `data/phase2/results_cpu.csv` and marks each model
`fits (< 1024 MB)` or `OOM`. Expected: yolo11n/yolo11n-seg/rfdetr-nano fit; sam2-tiny,
yolo-world, mask2former exceed 1 GB. A note records that the fitting models are additionally
confirmed to run on the live Streamlit Cloud free tier (human-observed), while the others OOM.

**5. Generators.**
- `benchmark/latex_tables.py`: add `distribution_table(dist_rows, label)` (per model×device:
  mean, std, P50, P90, P99 at 640) and `feasibility_table(perf_rows, label)` (per model: peak
  RSS, fits/OOM on 1 GB).
- `benchmark/plot.py`: add `plot_latency_boxplot(dist_rows, out_path)` — a box plot of the
  per-image latency samples per model (CPU; optionally CPU vs GPU as grouped boxes; log y since
  CPU/GPU span orders of magnitude).
- `benchmark/build_report_assets.py`: also emit the distribution table, the feasibility table,
  and the box-plot when the `results_dist_*` CSVs are present.

## Part B — Paper integration (report repo)

- **New Results subsection "Latency distribution across images"**: the distribution table
  (mean/std/P50/P90/P99, CPU and GPU) + the box-plot figure (`fig:latency-dist`), stating the
  protocol (100 COCO val2017 images at 640, one timed inference each after warmup).
- **New "Deployment feasibility on the ~1 GB tier" table**: per model, isolated peak RSS and
  fits/OOM on the Streamlit free tier — closing the P1 caveat with a concrete result.
- **Threats to validity**: replace the "single sample image" caveat with the 100-image
  distribution; keep the different-software-stacks caveat.
- **Abstract + Method**: mention that latency is a distribution over 100 images with reported
  percentiles.
- The existing perf/cold-warm/memory tables and the scaling figures are unchanged.

## Implementation decomposition (two plans, human GPU handoff between)

- **Plan A (benchmark repo):** 100-image fetch, percentile primitive, `run_dist`, the three
  generators, TDD, and the **CPU distribution run** producing `results_dist_cpu.csv`. Ends with
  the author running `run_dist --device cuda` on Colab to produce `results_dist_cuda.csv`.
- **Plan B (report repo):** once both distribution CSVs exist — generate the table/figure/1 GB
  table, write the Results subsection, adjust threats/abstract/method, rebuild.

## Out of scope — Future Work (the reviewer's other asks)

Deferred, not in Phase 2c: own accuracy/mAP evaluation on COCO, PQ/mIoU for segmentation,
ONNX/OpenVINO/quantization comparison, and a YOLO-World prompt-count study. Recorded in the
paper's Future Work.

## Success criteria

- `run_dist` produces `results_dist_{cpu,cuda}.csv` with 100 per-image latency rows per model,
  plus a hardware manifest; percentile stats computed from them.
- The paper gains a distribution table (mean/std/P50/P90/P99) + a box-plot + a 1 GB feasibility
  table; the "single sample image" threat is retired; build stays clean (0 unresolved refs).
- All new harness code is covered by tests consistent with the existing suite; the suite stays
  green.
