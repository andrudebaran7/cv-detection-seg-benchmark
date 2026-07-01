from __future__ import annotations

import statistics
import time
from typing import Callable

import psutil


def _cuda_sync_fn() -> Callable[[], None]:
    """Resolve a CUDA barrier ONCE (outside any timed region).

    CUDA kernels are asynchronous: ``fn()`` can return before the GPU has finished,
    so wall-clock timing must wait on the device to attribute the full compute cost.
    Returns ``torch.cuda.synchronize`` on a GPU host, else a no-op (CPU, or torch
    absent) so the timing code stays device-agnostic. Resolving it here avoids paying
    the import/availability check on every iteration inside the timed loop.
    """
    try:
        import torch
    except Exception:
        return lambda: None
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return lambda: None


def timeit_callable(fn: Callable[[], object], *, warmup: int = 5, iters: int = 50) -> dict:
    sync = _cuda_sync_fn()
    for _ in range(warmup):
        fn()
    sync()  # drain warmup GPU work so it can't bleed into the first timed sample
    samples_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync()  # wait for this call's GPU work before stopping the clock
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "n_iters": iters,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "std_ms": statistics.pstdev(samples_ms) if iters > 1 else 0.0,
    }


def time_first_call_ms(fn: Callable[[], object]) -> float:
    sync = _cuda_sync_fn()
    start = time.perf_counter()
    fn()
    sync()  # include the GPU work of the first (cold) call in the reading
    return (time.perf_counter() - start) * 1000.0


def peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def reset_peak_gpu() -> None:
    """Reset CUDA peak-memory tracking so the next measurement is isolated. No-op on CPU."""
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_mb() -> float:
    """Peak CUDA memory allocated (MB) since the last reset; 0.0 when no GPU is present."""
    try:
        import torch
    except Exception:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _percentile(sorted_samples, pct: float) -> float:
    """Linear-interpolation percentile (numpy default), pct in [0, 100]."""
    if not sorted_samples:
        raise ValueError("empty samples")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (len(sorted_samples) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return float(sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac)


def latency_stats(samples_ms) -> dict:
    """Distribution summary over per-image latency samples (ms)."""
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "n": n,
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered) if n > 1 else 0.0,
        "p50_ms": _percentile(ordered, 50),
        "p90_ms": _percentile(ordered, 90),
        "p99_ms": _percentile(ordered, 99),
    }


def time_per_image(predict_one: Callable[[object], object], images, *, warmup: int = 5) -> list:
    """Time one inference per image (ms) after a global warmup on the first image.

    ``predict_one(img)`` runs one inference. CUDA work is drained per call so wall-clock
    timing attributes the full GPU cost (mirrors ``timeit_callable``).
    """
    sync = _cuda_sync_fn()
    if images:
        for _ in range(warmup):
            predict_one(images[0])
        sync()
    samples_ms = []
    for img in images:
        start = time.perf_counter()
        predict_one(img)
        sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms
