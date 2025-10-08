"""Scalability helpers for throughput and latency experiments."""
from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Callable, Dict, Any, Iterable

import numpy as np


def measure_latency(
    fn: Callable[[np.ndarray], np.ndarray],
    batch: np.ndarray,
    repeats: int = 10,
) -> Dict[str, Any]:
    """Return latency distribution stats (seconds)."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(batch)
        timings.append(time.perf_counter() - start)
    sample = timings if len(timings) > 1 else timings + [timings[0]]
    return {
        "mean": float(mean(sample)),
        "std": float(stdev(sample)),
        "min": float(min(sample)),
        "max": float(max(sample)),
        "repeats": repeats,
    }


def benchmark_throughput(
    fn: Callable[[np.ndarray], np.ndarray],
    loader: Iterable[np.ndarray],
    warmup: int = 1,
) -> Dict[str, Any]:
    """Iterate through batches and compute total items processed per second."""
    total_items = 0
    total_time = 0.0

    for i, batch in enumerate(loader):
        if i < warmup:
            fn(batch)
            continue
        start = time.perf_counter()
        output = fn(batch)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_items += batch.shape[0]
    throughput = (total_items / total_time) if total_time else 0.0
    return {
        "total_items": int(total_items),
        "total_time": float(total_time),
        "throughput": float(throughput),
        "warmup_batches": warmup,
    }
