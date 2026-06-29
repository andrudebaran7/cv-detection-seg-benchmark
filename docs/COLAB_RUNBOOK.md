# Phase 2 — running the GPU pass on Colab

The measurement harness (`benchmark/`) is device-portable: the same command that
produced `data/phase2/results_cpu.csv` locally produces the GPU CSV on Colab. Plan 2b
(paper integration) merges both into the report's tables and figures.

1. Open a GPU runtime: **Runtime → Change runtime type → GPU**.
2. Clone the repo and install deps (Colab ships a CUDA `torch`, so install the dev deps
   without the CPU-only torch pin — install the harness extras directly):
   ```
   !git clone <repo-url>
   %cd cv-detection-seg-benchmark
   !pip install psutil matplotlib ultralytics rfdetr supervision transformers ftfy
   !pip install git+https://github.com/ultralytics/CLIP.git
   ```
3. Fetch the image set:
   ```
   !python -m benchmark.fetch_images
   ```
4. Run the campaign on GPU (use the full iteration count for the real numbers):
   ```
   !python -m benchmark.run --device cuda --models all --iters 50 --warmup 5 --out data/phase2
   ```
   The run is resilient: if a single model fails (e.g. a transient HuggingFace download),
   it is skipped with a `[skip] <model>: ...` line and the others still produce data — re-run
   just that model with `--models <key>` afterwards.
5. Download `data/phase2/results_cuda.csv` and `data/phase2/manifest_cuda.json`, then commit
   them next to the CPU files (`data/phase2/results_cpu.csv`, `manifest_cpu.json`).

Once both `results_cpu.csv` and `results_cuda.csv` exist, Plan 2b generates the final
figures and writes the Results tables in the report.
