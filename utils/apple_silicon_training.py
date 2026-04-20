"""Apple Silicon training utilities for Notebook 08: Training on Apple Silicon.

Provides gradient accumulation, mixed precision training, memory profiling,
memory budget calculation, OOM recovery, and mx.compile() benchmarks — all MLX.

**Validates: Requirements 8.1–8.7**
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_memory_mb() -> float:
    """Return active Metal GPU memory in MB, or 0.0 if unavailable."""
    try:
        return mx.metal.get_active_memory() / 1e6
    except Exception:
        return 0.0


def _get_peak_memory_mb() -> float:
    """Return peak Metal GPU memory in MB, or 0.0 if unavailable."""
    try:
        return mx.metal.get_peak_memory() / 1e6
    except Exception:
        return 0.0


def _get_cache_memory_mb() -> float:
    """Return cache Metal GPU memory in MB, or 0.0 if unavailable."""
    try:
        return mx.metal.get_cache_memory() / 1e6
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 10.1  GradientAccumulator
# ---------------------------------------------------------------------------

class GradientAccumulator:
    """Accumulate gradients over K micro-batches to simulate a larger batch.

    💡 effective_batch_size = micro_batch_size × accum_steps
    ⚡ Memory usage stays at micro_batch_size level while getting large-batch gradients.

    🎯 Interview tip: gradient accumulation is essential when GPU memory limits
    batch size but you need large effective batches for stable training.
    """

    @staticmethod
    def accumulate(
        model: nn.Module,
        loss_fn: Callable,
        data_batches: list,
        accum_steps: int | None = None,
    ) -> tuple[float, dict]:
        """Accumulate gradients over micro-batches.

        Args:
            model: The nn.Module to compute gradients for.
            loss_fn: Callable(model, batch) -> scalar loss.
            data_batches: List of micro-batches to accumulate over.
            accum_steps: Number of micro-batches (defaults to len(data_batches)).

        Returns:
            (average_loss, accumulated_gradients) where gradients are averaged
            over all micro-batches — equivalent to a single large-batch step.
        """
        if accum_steps is None:
            accum_steps = len(data_batches)
        assert accum_steps > 0, "accum_steps must be positive"
        assert len(data_batches) >= accum_steps, (
            f"Need at least {accum_steps} micro-batches, got {len(data_batches)}"
        )

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        total_loss = 0.0
        accumulated_grads = None

        for i in range(accum_steps):
            batch = data_batches[i]
            loss_val, grads = loss_and_grad_fn(model, batch)
            mx.eval(loss_val, grads)
            total_loss += loss_val.item()

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                # Sum gradients element-wise
                acc_flat = tree_flatten(accumulated_grads)
                new_flat = tree_flatten(grads)
                summed = []
                for (k, a), (_, g) in zip(acc_flat, new_flat):
                    summed.append((k, a + g))
                accumulated_grads = tree_unflatten(summed)
                mx.eval(accumulated_grads)

        # ⚠️ Average gradients over micro-batches (critical for equivalence!)
        avg_flat = tree_flatten(accumulated_grads)
        averaged = [(k, g / accum_steps) for k, g in avg_flat]
        accumulated_grads = tree_unflatten(averaged)
        mx.eval(accumulated_grads)

        avg_loss = total_loss / accum_steps
        return avg_loss, accumulated_grads


# ---------------------------------------------------------------------------
# 10.3  MixedPrecisionTrainer
# ---------------------------------------------------------------------------

class MixedPrecisionTrainer:
    """Compare float32 vs bfloat16 training on speed, memory, and loss stability.

    ⚡ bfloat16 has the same exponent range as float32 but fewer mantissa bits —
    more stable for training than float16 while using half the memory.

    🎯 Interview tip: most modern LLM training uses bfloat16 (Google TPUs, Apple MLX).
    """

    @staticmethod
    def train_step_f32(
        model: nn.Module,
        loss_fn: Callable,
        batch: Any,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float]:
        """Single training step in float32.

        Returns:
            (loss_value, elapsed_seconds)
        """
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        t0 = time.perf_counter()
        loss_val, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_val)
        t1 = time.perf_counter()
        return loss_val.item(), t1 - t0

    @staticmethod
    def train_step_bf16(
        model: nn.Module,
        loss_fn: Callable,
        batch: Any,
        optimizer: optim.Optimizer,
    ) -> tuple[float, float]:
        """Single training step in bfloat16.

        ⚡ Casts inputs to bfloat16 for the forward/backward pass.

        Returns:
            (loss_value, elapsed_seconds)
        """
        def bf16_loss_fn(mdl, b):
            x, y = b
            # Only cast FLOATING-point inputs to bf16. Integer token-id inputs
            # must stay integral so downstream nn.Embedding lookups work — the
            # pre-2025 version of this helper erroneously cast all inputs.
            if mx.issubdtype(x.dtype, mx.floating):
                x_cast = x.astype(mx.bfloat16)
            else:
                x_cast = x
            logits = mdl(x_cast)
            logits_f32 = logits.astype(mx.float32)
            return nn.losses.cross_entropy(logits_f32, y, reduction="mean")

        loss_and_grad_fn = nn.value_and_grad(model, bf16_loss_fn)
        t0 = time.perf_counter()
        loss_val, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_val)
        t1 = time.perf_counter()
        return loss_val.item(), t1 - t0

    @staticmethod
    def compare_dtypes(
        model_factory: Callable,
        loss_fn: Callable,
        batch: Any,
        n_steps: int = 5,
    ) -> dict:
        """Compare float32 vs bfloat16 training over n_steps.

        Args:
            model_factory: Callable that returns a fresh model instance.
            loss_fn: Callable(model, batch) -> scalar loss.
            batch: A (x, y) tuple for training.
            n_steps: Number of steps to benchmark.

        Returns:
            Dict with 'float32' and 'bfloat16' sub-dicts containing
            'losses', 'times', 'avg_time', 'memory_mb'.
        """
        results = {}

        for dtype_name in ["float32", "bfloat16"]:
            model = model_factory()
            mx.eval(model.parameters())
            opt = optim.Adam(learning_rate=1e-3)

            mem_before = _get_active_memory_mb()
            losses = []
            times = []

            train_fn = (
                MixedPrecisionTrainer.train_step_f32
                if dtype_name == "float32"
                else MixedPrecisionTrainer.train_step_bf16
            )

            for _ in range(n_steps):
                loss_val, elapsed = train_fn(model, loss_fn, batch, opt)
                losses.append(loss_val)
                times.append(elapsed)

            mem_after = _get_active_memory_mb()
            results[dtype_name] = {
                "losses": losses,
                "times": times,
                "avg_time": sum(times) / len(times),
                "memory_mb": mem_after - mem_before,
            }

        return results


# ---------------------------------------------------------------------------
# 10.4  MemoryProfiler
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """A single memory measurement at a training phase."""
    phase: str
    active_mb: float
    peak_mb: float
    cache_mb: float
    timestamp: float = 0.0


@dataclass
class MemoryTimeline:
    """Timeline of memory snapshots across a training step."""
    snapshots: list = field(default_factory=list)

    def add(self, phase: str) -> MemorySnapshot:
        snap = MemorySnapshot(
            phase=phase,
            active_mb=_get_active_memory_mb(),
            peak_mb=_get_peak_memory_mb(),
            cache_mb=_get_cache_memory_mb(),
            timestamp=time.perf_counter(),
        )
        self.snapshots.append(snap)
        return snap

    @property
    def phases(self) -> list[str]:
        return [s.phase for s in self.snapshots]

    @property
    def active_mbs(self) -> list[float]:
        return [s.active_mb for s in self.snapshots]

    @property
    def peak_mbs(self) -> list[float]:
        return [s.peak_mb for s in self.snapshots]


class MemoryProfiler:
    """Profile memory usage across training step phases.

    💡 Tracks memory at: baseline → forward → loss → backward → optimizer → cleanup.
    """

    @staticmethod
    def profile_training_step(
        model: nn.Module,
        loss_fn: Callable,
        batch: Any,
        optimizer: optim.Optimizer,
    ) -> MemoryTimeline:
        """Profile a single training step, recording memory at each phase.

        Returns:
            MemoryTimeline with snapshots at each phase.
        """
        timeline = MemoryTimeline()

        # Phase 1: Baseline
        timeline.add("baseline")

        # Phase 2: Forward pass
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad_fn(model, batch)
        mx.eval(loss_val)
        timeline.add("forward+backward")

        # Phase 3: Optimizer update
        mx.eval(grads)
        timeline.add("grads_materialized")

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        timeline.add("optimizer_update")

        # Phase 4: Cleanup (delete grads reference)
        del grads
        timeline.add("cleanup")

        return timeline

    @staticmethod
    def plot_memory_timeline(timeline: MemoryTimeline):
        """Plot memory usage across training phases.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        phases = timeline.phases
        active = timeline.active_mbs
        peak = timeline.peak_mbs

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(phases))

        ax.bar(x, active, width=0.4, label="Active Memory (MB)",
               color="#4A90D9", alpha=0.8, align="center")
        ax.plot(x, peak, "r--o", label="Peak Memory (MB)", linewidth=2)

        ax.set_xticks(list(x))
        ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Memory (MB)")
        ax.set_title("⚡ Memory Profile: Training Step Phases")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 10.5  MemoryBudgetCalculator
# ---------------------------------------------------------------------------

class MemoryBudgetCalculator:
    """Compute memory breakdown for transformer training.

    💡 Components: parameters, gradients, optimizer states, activations, KV-cache.
    🎯 Interview tip: Adam stores 2 extra copies (m, v) per parameter in float32.

    ⚠️ Activations scale with batch_size × seq_len — the main OOM culprit.
    """

    @staticmethod
    def compute_budget(
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        batch_size: int,
        seq_len: int,
        vocab_size: int = 32000,
        dtype_bytes: int = 2,
    ) -> dict:
        """Compute memory budget breakdown in MB.

        Args:
            d_model: Model hidden dimension.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            d_ff: FFN hidden dimension.
            batch_size: Training batch size.
            seq_len: Sequence length.
            vocab_size: Vocabulary size.
            dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32).

        Returns:
            Dict with 'params_mb', 'grads_mb', 'optimizer_mb',
            'activations_mb', 'kv_cache_mb', 'total_mb', and 'components'.
        """
        # --- Parameter count ---
        # Attention: Q, K, V, O projections per layer
        attn_params = n_layers * (4 * d_model * d_model)
        # FFN: up + down projections per layer
        ffn_params = n_layers * (2 * d_model * d_ff)
        # Normalization: 2 norms per layer × d_model
        norm_params = n_layers * (2 * d_model)
        # Embedding + output head
        embed_params = vocab_size * d_model
        total_params = attn_params + ffn_params + norm_params + embed_params

        # --- Memory components ---
        params_mb = total_params * dtype_bytes / 1e6
        grads_mb = total_params * dtype_bytes / 1e6  # Same size as params
        # Adam: 2 states (m, v) in float32
        optimizer_mb = 2 * total_params * 4 / 1e6
        # Activations: per-layer, per-token hidden states
        activations_mb = (
            batch_size * seq_len * d_model * n_layers * dtype_bytes / 1e6
        )
        # KV-cache (inference only, included for completeness)
        head_dim = d_model // n_heads
        kv_cache_mb = (
            2 * n_layers * batch_size * seq_len * d_model * dtype_bytes / 1e6
        )

        total_mb = params_mb + grads_mb + optimizer_mb + activations_mb

        return {
            "params_mb": params_mb,
            "grads_mb": grads_mb,
            "optimizer_mb": optimizer_mb,
            "activations_mb": activations_mb,
            "kv_cache_mb": kv_cache_mb,
            "total_mb": total_mb,
            "total_params": total_params,
            "components": {
                "Parameters": params_mb,
                "Gradients": grads_mb,
                "Optimizer (Adam m+v)": optimizer_mb,
                "Activations": activations_mb,
            },
        }

    @staticmethod
    def plot_stacked_bar(budget: dict, title: str = "Memory Budget Breakdown"):
        """Plot a stacked bar chart of memory components.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        components = budget["components"]
        labels = list(components.keys())
        values = list(components.values())
        colors = ["#4A90D9", "#7B68EE", "#FF6B6B", "#4ECDC4"]

        fig, ax = plt.subplots(figsize=(8, 5))

        bottom = 0
        bars = []
        for label, val, color in zip(labels, values, colors):
            bar = ax.bar("Training", val, bottom=bottom, label=label,
                         color=color, alpha=0.85, width=0.5)
            bars.append(bar)
            bottom += val

        # Add KV-cache as separate bar for inference
        kv_mb = budget["kv_cache_mb"]
        ax.bar("Inference\n(KV-cache)", kv_mb, color="#FFD93D",
               alpha=0.85, width=0.5, label="KV-Cache")

        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"🎯 {title}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate total
        ax.annotate(
            f"Total: {budget['total_mb']:.1f} MB\n({budget['total_mb']/1024:.2f} GB)",
            xy=(0, bottom), xytext=(0.3, bottom * 0.9),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 10.7  OOM Recovery & mx.compile() Benchmarks
# ---------------------------------------------------------------------------

def auto_reduce_batch_size(
    fn: Callable,
    initial_batch_size: int,
    min_batch_size: int = 1,
) -> int:
    """Automatically reduce batch size until fn succeeds without OOM.

    ⚠️ Catches RuntimeError from MLX Metal OOM and halves batch size.

    Args:
        fn: Callable(batch_size) -> None. Should run a training step.
        initial_batch_size: Starting batch size.
        min_batch_size: Minimum batch size to try.

    Returns:
        The largest working batch size.
    """
    batch_size = initial_batch_size
    while batch_size >= min_batch_size:
        try:
            fn(batch_size)
            return batch_size
        except (RuntimeError, MemoryError) as e:
            err_msg = str(e).lower()
            if "memory" in err_msg or "oom" in err_msg or "allocation" in err_msg:
                print(f"⚠️ OOM at batch_size={batch_size}, reducing to {batch_size // 2}")
                batch_size = batch_size // 2
            else:
                raise
    raise RuntimeError(
        f"Cannot fit even batch_size={min_batch_size}. Consider reducing model size."
    )


def compile_benchmark(
    fn: Callable,
    x: mx.array,
    n_iters: int = 20,
    warmup: int = 5,
) -> dict:
    """Benchmark a function with and without mx.compile().

    ⚡ mx.compile() fuses operations and reduces kernel launch overhead.

    Args:
        fn: Function to benchmark (takes a single mx.array argument).
        x: Input tensor.
        n_iters: Number of timed iterations.
        warmup: Number of warmup iterations.

    Returns:
        Dict with 'uncompiled_ms', 'compiled_ms', 'speedup'.
    """
    compiled_fn = mx.compile(fn)

    # Warmup both
    for _ in range(warmup):
        mx.eval(fn(x))
        mx.eval(compiled_fn(x))

    # Benchmark uncompiled
    t0 = time.perf_counter()
    for _ in range(n_iters):
        mx.eval(fn(x))
    t1 = time.perf_counter()

    # Benchmark compiled
    for _ in range(n_iters):
        mx.eval(compiled_fn(x))
    t2 = time.perf_counter()

    uncompiled_ms = (t1 - t0) / n_iters * 1000
    compiled_ms = (t2 - t1) / n_iters * 1000
    speedup = uncompiled_ms / compiled_ms if compiled_ms > 0 else float("inf")

    return {
        "uncompiled_ms": uncompiled_ms,
        "compiled_ms": compiled_ms,
        "speedup": speedup,
    }
