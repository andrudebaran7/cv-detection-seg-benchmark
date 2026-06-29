from __future__ import annotations

import statistics
import time
from typing import Callable

import psutil


def timeit_callable(fn: Callable[[], object], *, warmup: int = 5, iters: int = 50) -> dict:
    for _ in range(warmup):
        fn()
    samples_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "n_iters": iters,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "std_ms": statistics.pstdev(samples_ms) if iters > 1 else 0.0,
    }


def time_first_call_ms(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
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
