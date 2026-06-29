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
