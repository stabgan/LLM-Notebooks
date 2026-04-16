"""Timing and memory measurement helpers for Apple Silicon ML."""

from __future__ import annotations

import time
from typing import Any, Callable


def time_function(fn: Callable, *args: Any, warmup: int = 1, repeats: int = 5, **kwargs: Any) -> dict:
    """Time a function with warmup runs and return statistics.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments forwarded to *fn*.
        warmup: Number of warmup calls (not timed).
        repeats: Number of timed calls.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        Dict with keys: mean_ms, min_ms, max_ms, repeats, result (last call's return value).
    """
    import mlx.core as mx

    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        mx.eval(result) if hasattr(result, "shape") else None

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        # Force evaluation for lazy MLX arrays
        if hasattr(result, "shape"):
            mx.eval(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "repeats": repeats,
        "result": result,
    }


def memory_snapshot() -> dict:
    """Return current Metal GPU memory usage.

    Uses ``mx.metal.get_active_memory()`` when available.

    Returns:
        Dict with keys: active_bytes, peak_bytes, cache_bytes, active_mb.
        All values are 0 when Metal is not available.
    """
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return {"active_bytes": 0, "peak_bytes": 0, "cache_bytes": 0, "active_mb": 0.0}

        # Prefer new top-level API (MLX ≥ 0.31), fall back to mx.metal.*
        active = mx.get_active_memory() if hasattr(mx, "get_active_memory") else mx.metal.get_active_memory()
        peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
        cache = mx.get_cache_memory() if hasattr(mx, "get_cache_memory") else mx.metal.get_cache_memory()
        return {
            "active_bytes": active,
            "peak_bytes": peak,
            "cache_bytes": cache,
            "active_mb": round(active / (1024 ** 2), 2),
        }
    except Exception:
        return {"active_bytes": 0, "peak_bytes": 0, "cache_bytes": 0, "active_mb": 0.0}


def estimate_model_memory(n_params: int, dtype_bytes: int = 2) -> dict:
    """Estimate memory required for a model's weights and optimizer state.

    Args:
        n_params: Total number of model parameters.
        dtype_bytes: Bytes per parameter (2 for float16/bfloat16, 4 for float32).

    Returns:
        Dict with keys: weights_mb, optimizer_mb (Adam ≈ 2× weights), total_mb.
    """
    weights = n_params * dtype_bytes
    # Adam stores first and second moments, each the size of the weights
    optimizer = 2 * weights
    total = weights + optimizer
    return {
        "weights_mb": round(weights / (1024 ** 2), 2),
        "optimizer_mb": round(optimizer / (1024 ** 2), 2),
        "total_mb": round(total / (1024 ** 2), 2),
    }

