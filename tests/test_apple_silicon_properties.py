"""Property-based tests for Apple Silicon training utilities (Notebook 08).

**Validates: Requirements 8.1, 8.4**

Property 21: Gradient Accumulation Equivalence — accumulated gradients match
             large-batch gradients within floating-point tolerance.
Property 22: Memory Budget Calculator Consistency — sum of components equals total.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from utils.apple_silicon_training import (
    GradientAccumulator,
    MemoryBudgetCalculator,
)


# ---------------------------------------------------------------------------
# Shared: tiny model for gradient tests
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    """Minimal MLP for testing gradient accumulation."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Property 21: Gradient Accumulation Equivalence
# ---------------------------------------------------------------------------

@given(
    d_in=st.sampled_from([4, 8]),
    d_hidden=st.sampled_from([8, 16]),
    d_out=st.sampled_from([4, 8]),
    micro_batch_size=st.sampled_from([2, 4]),
    accum_steps=st.sampled_from([2, 4]),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=30)
def test_gradient_accumulation_equivalence(
    d_in, d_hidden, d_out, micro_batch_size, accum_steps, seed,
):
    """**Validates: Requirement 8.1**

    Property 21: Training with batch_size B and accum_steps 1 produces
    gradients equivalent to training with batch_size B/K and accum_steps K.

    We create a full batch of size (micro_batch_size * accum_steps), compute
    gradients in one shot, then compare with accumulated micro-batch gradients.
    """
    mx.random.seed(seed)
    full_batch_size = micro_batch_size * accum_steps

    # Create deterministic data
    x_full = mx.random.normal(shape=(full_batch_size, d_in))
    y_full = mx.random.normal(shape=(full_batch_size, d_out))
    mx.eval(x_full, y_full)

    # --- Model A: single large-batch gradient ---
    model_a = _TinyMLP(d_in, d_hidden, d_out)
    mx.eval(model_a.parameters())

    # Copy weights to model B so they start identical
    model_b = _TinyMLP(d_in, d_hidden, d_out)
    model_b.load_weights(list(tree_flatten(model_a.parameters())))
    mx.eval(model_b.parameters())

    def loss_fn(mdl, batch):
        x, y = batch
        pred = mdl(x)
        return mx.mean((pred - y) ** 2)

    # Full-batch gradient
    loss_and_grad = nn.value_and_grad(model_a, loss_fn)
    full_loss, full_grads = loss_and_grad(model_a, (x_full, y_full))
    mx.eval(full_loss, full_grads)

    # --- Model B: accumulated micro-batch gradients ---
    micro_batches = []
    for i in range(accum_steps):
        start = i * micro_batch_size
        end = start + micro_batch_size
        micro_batches.append((x_full[start:end], y_full[start:end]))

    accum_loss, accum_grads = GradientAccumulator.accumulate(
        model_b, loss_fn, micro_batches, accum_steps=accum_steps,
    )

    # --- Compare gradients ---
    full_flat = tree_flatten(full_grads)
    accum_flat = tree_flatten(accum_grads)

    assert len(full_flat) == len(accum_flat), (
        f"Gradient tree mismatch: {len(full_flat)} vs {len(accum_flat)}"
    )

    for (k_f, g_f), (k_a, g_a) in zip(full_flat, accum_flat):
        mx.eval(g_f, g_a)
        np.testing.assert_allclose(
            np.array(g_f),
            np.array(g_a),
            atol=1e-4,
            rtol=1e-3,
            err_msg=(
                f"Gradient mismatch for {k_f}: "
                f"max diff = {np.max(np.abs(np.array(g_f) - np.array(g_a)))}"
            ),
        )


# ---------------------------------------------------------------------------
# Property 22: Memory Budget Calculator Consistency
# ---------------------------------------------------------------------------

@given(
    d_model=st.sampled_from([64, 128, 256, 512]),
    n_layers=st.integers(min_value=1, max_value=12),
    n_heads=st.sampled_from([2, 4, 8]),
    d_ff_mult=st.sampled_from([2, 4]),
    batch_size=st.integers(min_value=1, max_value=32),
    seq_len=st.sampled_from([64, 128, 256, 512]),
    dtype_bytes=st.sampled_from([2, 4]),
)
@settings(deadline=None, max_examples=100)
def test_memory_budget_consistency(
    d_model, n_layers, n_heads, d_ff_mult, batch_size, seq_len, dtype_bytes,
):
    """**Validates: Requirement 8.4**

    Property 22: The sum of memory components (parameters, gradients,
    optimizer states, activations) equals the reported total memory.
    """
    assume(d_model % n_heads == 0)
    d_ff = d_model * d_ff_mult

    budget = MemoryBudgetCalculator.compute_budget(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
    )

    # Sum of components must equal total
    component_sum = (
        budget["params_mb"]
        + budget["grads_mb"]
        + budget["optimizer_mb"]
        + budget["activations_mb"]
    )

    assert abs(component_sum - budget["total_mb"]) < 1e-6, (
        f"Component sum ({component_sum:.6f}) != total ({budget['total_mb']:.6f}). "
        f"Diff: {abs(component_sum - budget['total_mb']):.10f}"
    )

    # All components must be non-negative
    assert budget["params_mb"] >= 0, f"Negative params: {budget['params_mb']}"
    assert budget["grads_mb"] >= 0, f"Negative grads: {budget['grads_mb']}"
    assert budget["optimizer_mb"] >= 0, f"Negative optimizer: {budget['optimizer_mb']}"
    assert budget["activations_mb"] >= 0, f"Negative activations: {budget['activations_mb']}"
    assert budget["kv_cache_mb"] >= 0, f"Negative kv_cache: {budget['kv_cache_mb']}"
    assert budget["total_mb"] > 0, f"Total must be positive: {budget['total_mb']}"

    # Params == Grads (same size)
    assert abs(budget["params_mb"] - budget["grads_mb"]) < 1e-6, (
        f"Params ({budget['params_mb']}) should equal grads ({budget['grads_mb']})"
    )

    # Optimizer should be larger than params when dtype_bytes < 4
    # (Adam stores in float32 = 4 bytes, 2 states)
    if dtype_bytes < 4:
        assert budget["optimizer_mb"] > budget["params_mb"], (
            f"Optimizer ({budget['optimizer_mb']}) should exceed params "
            f"({budget['params_mb']}) for dtype_bytes={dtype_bytes}"
        )

    # Components dict should also sum to total
    components_dict_sum = sum(budget["components"].values())
    assert abs(components_dict_sum - budget["total_mb"]) < 1e-6, (
        f"Components dict sum ({components_dict_sum}) != total ({budget['total_mb']})"
    )
