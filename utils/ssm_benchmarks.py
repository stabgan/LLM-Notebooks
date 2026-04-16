"""
Attention vs SSM benchmark utilities for Notebook 16.

Provides timing benchmarks, memory scaling plots, and quality
comparison visualizations for attention vs Mamba-style SSMs.
All implementations use MLX exclusively.
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn

from utils.ssm import SSMConfig, MambaBlock


def _simple_attention(Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
    """Standard scaled dot-product attention in MLX.

    Parameters
    ----------
    Q, K, V : mx.array, shape [batch, seq, d_model]

    Returns
    -------
    mx.array, shape [batch, seq, d_model]
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(d_k)  # [B, S, S]
    weights = mx.softmax(scores, axis=-1)
    return weights @ V  # [B, S, d_model]


def benchmark_attention_vs_ssm(
    d_model: int = 64,
    seq_lengths: list[int] | None = None,
    batch_size: int = 2,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> dict:
    """Time both attention and MambaBlock at each sequence length.

    Parameters
    ----------
    d_model : int
        Model dimension.
    seq_lengths : list[int], optional
        Sequence lengths to benchmark. Defaults to [64, 128, 256, 512, 1024].
    batch_size : int
        Batch size for benchmarks.
    n_warmup : int
        Number of warmup runs (not timed).
    n_runs : int
        Number of timed runs to average.

    Returns
    -------
    dict with keys:
        - seq_lengths: list[int]
        - attn_times_ms: list[float]
        - ssm_times_ms: list[float]
        - attn_memory_bytes: list[int]  (theoretical)
        - ssm_memory_bytes: list[int]   (theoretical)
        - d_model: int
        - batch_size: int
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024]

    config = SSMConfig(d_model=d_model, d_state=16)
    mamba = MambaBlock(config)

    attn_times = []
    ssm_times = []
    attn_mem = []
    ssm_mem = []

    for seq_len in seq_lengths:
        x = mx.random.normal((batch_size, seq_len, d_model))
        mx.eval(x)

        # --- Warmup attention ---
        for _ in range(n_warmup):
            Q = K = V = x
            out = _simple_attention(Q, K, V)
            mx.eval(out)

        # --- Time attention ---
        times = []
        for _ in range(n_runs):
            Q = K = V = x
            t0 = time.perf_counter()
            out = _simple_attention(Q, K, V)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        attn_times.append(float(np.median(times)))

        # --- Warmup SSM ---
        for _ in range(n_warmup):
            out = mamba(x)
            mx.eval(out)

        # --- Time SSM ---
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = mamba(x)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        ssm_times.append(float(np.median(times)))

        # Theoretical memory: attention materializes n×n, SSM keeps d_inner×d_state
        attn_mem.append(batch_size * seq_len * seq_len * 4)  # float32 attn matrix
        ssm_mem.append(batch_size * config.d_inner * config.d_state * 4)

    return {
        "seq_lengths": seq_lengths,
        "attn_times_ms": attn_times,
        "ssm_times_ms": ssm_times,
        "attn_memory_bytes": attn_mem,
        "ssm_memory_bytes": ssm_mem,
        "d_model": d_model,
        "batch_size": batch_size,
    }


def plot_benchmark_results(results: dict) -> plt.Figure:
    """Plot timing and memory comparison from benchmark results.

    Parameters
    ----------
    results : dict
        Output from benchmark_attention_vs_ssm().

    Returns
    -------
    matplotlib.figure.Figure
    """
    seq_lengths = results["seq_lengths"]
    attn_t = results["attn_times_ms"]
    ssm_t = results["ssm_times_ms"]
    attn_m = results["attn_memory_bytes"]
    ssm_m = results["ssm_memory_bytes"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Wall-clock timing ---
    ax = axes[0]
    ax.plot(seq_lengths, attn_t, "o-", color="#e74c3c", linewidth=2, label="Attention")
    ax.plot(seq_lengths, ssm_t, "s-", color="#2ecc71", linewidth=2, label="MambaBlock")
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("⚡ Wall-Clock Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Theoretical memory ---
    ax = axes[1]
    ax.semilogy(seq_lengths, attn_m, "o-", color="#e74c3c", linewidth=2, label="Attention O(n²)")
    ax.semilogy(seq_lengths, ssm_m, "s-", color="#2ecc71", linewidth=2, label="SSM O(1)")
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Memory (bytes, log)", fontsize=11)
    ax.set_title("💾 Memory Scaling", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Speedup ratio ---
    ax = axes[2]
    ratios = [a / s if s > 0 else 1.0 for a, s in zip(attn_t, ssm_t)]
    colors = ["#2ecc71" if r > 1 else "#e74c3c" for r in ratios]
    bars = ax.bar([str(s) for s in seq_lengths], ratios, color=colors, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Speedup (Attn time / SSM time)", fontsize=11)
    ax.set_title("🎯 Relative Speedup", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add ratio labels on bars
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ratio:.2f}×",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    fig.suptitle(
        f"Attention vs MambaBlock (d_model={results['d_model']}, batch={results['batch_size']})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_quality_comparison() -> plt.Figure:
    """Create a visual comparison of attention vs SSM quality tradeoffs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    categories = [
        "Short-range\n(≤512)",
        "Medium-range\n(512–4K)",
        "Long-range\n(4K–128K)",
        "In-context\nlearning",
        "Retrieval /\ncopy tasks",
        "Training\nparallelism",
    ]

    # Scores: 0–10 scale (qualitative, based on literature)
    attn_scores = [9, 8, 4, 9, 9, 8]
    ssm_scores = [7, 7, 9, 6, 5, 7]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width / 2, attn_scores, width, label="Attention",
                   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ssm_scores, width, label="SSM (Mamba)",
                   color="#2ecc71", alpha=0.85)

    ax.set_ylabel("Capability Score (qualitative)", fontsize=11)
    ax.set_title("🎯 Attention vs SSM: Quality Tradeoffs", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add score labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    # Annotations for key insights
    ax.annotate("SSM wins\nfor long seqs",
                xy=(2 + width / 2, 9), xytext=(2 + width / 2 + 0.6, 10.2),
                fontsize=9, color="#2ecc71", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2ecc71"))
    ax.annotate("Attention wins\nfor retrieval",
                xy=(4 - width / 2, 9), xytext=(4 - width / 2 - 1.2, 10.2),
                fontsize=9, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    plt.tight_layout()
    return fig
