"""Property-based tests for GPT training pipeline (Notebook 07).

**Validates: Requirements 7.2, 7.3, 7.4, 7.5**

Property 18: Cosine LR Schedule — LR matches formula for any step after warmup.
Property 19: Gradient Clipping Bound — global gradient norm ≤ clip threshold.
Property 20: Checkpoint Determinism — save/load produces identical model outputs.
"""

import math
import os
import shutil
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from utils.training import (
    cosine_lr_schedule,
    clip_grad_norm,
    CheckpointManager,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_positive_float = st.floats(min_value=1e-7, max_value=1.0, allow_nan=False, allow_infinity=False)
_lr_float = st.floats(min_value=1e-8, max_value=1e-1, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 18: Cosine LR Schedule
# ---------------------------------------------------------------------------

@given(
    step=st.integers(min_value=0, max_value=10000),
    warmup_steps=st.integers(min_value=0, max_value=500),
    max_steps=st.integers(min_value=100, max_value=10000),
    max_lr=st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False),
    min_lr=st.floats(min_value=1e-8, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cosine_lr_schedule(step, warmup_steps, max_steps, max_lr, min_lr):
    """**Validates: Requirement 7.2**

    Property 18: For any step after warmup, the learning rate equals
    min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × step / max_steps)).

    During warmup the LR linearly ramps from 0 to max_lr.
    """
    assume(min_lr < max_lr)
    assume(max_steps > 0)

    lr = cosine_lr_schedule(step, warmup_steps, max_steps, max_lr, min_lr)

    # LR must always be finite and non-negative
    assert math.isfinite(lr), f"LR is not finite: {lr}"
    assert lr >= 0, f"LR is negative: {lr}"

    if step < warmup_steps and warmup_steps > 0:
        # Linear warmup phase
        expected = max_lr * (step / warmup_steps)
        assert abs(lr - expected) < 1e-10, (
            f"Warmup LR mismatch: got {lr}, expected {expected} "
            f"(step={step}, warmup={warmup_steps})"
        )
    else:
        # Cosine decay phase — verify formula
        expected = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + math.cos(math.pi * step / max_steps)
        )
        assert abs(lr - expected) < 1e-10, (
            f"Cosine LR mismatch: got {lr}, expected {expected} "
            f"(step={step}, max_steps={max_steps})"
        )


# ---------------------------------------------------------------------------
# Property 19: Gradient Clipping Bound
# ---------------------------------------------------------------------------

@given(
    n_tensors=st.integers(min_value=1, max_value=5),
    tensor_size=st.integers(min_value=2, max_value=64),
    max_norm=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_gradient_clipping_bound(n_tensors, tensor_size, max_norm, scale):
    """**Validates: Requirement 7.3**

    Property 19: After gradient clipping, the global gradient norm is
    ≤ the clip threshold (max_norm).
    """
    # Build a fake gradient tree (list of named arrays)
    grads_list = []
    for i in range(n_tensors):
        g = mx.random.normal(shape=(tensor_size,)) * scale
        mx.eval(g)
        grads_list.append((f"layer_{i}.weight", g))

    grads = tree_unflatten(grads_list)
    clipped, original_norm = clip_grad_norm(grads, max_norm)

    # Compute clipped global norm
    flat_clipped = tree_flatten(clipped)
    sum_sq = mx.array(0.0)
    for _, g in flat_clipped:
        sum_sq = sum_sq + mx.sum(g * g)
    clipped_norm = mx.sqrt(sum_sq)
    mx.eval(clipped_norm)
    clipped_norm_val = clipped_norm.item()

    assert clipped_norm_val <= max_norm + 1e-5, (
        f"Clipped norm ({clipped_norm_val:.6f}) exceeds max_norm ({max_norm:.6f}). "
        f"Original norm was {original_norm:.6f}."
    )

    # If original was within budget, grads should be unchanged
    if original_norm <= max_norm:
        for (_, g_orig), (_, g_clip) in zip(
            tree_flatten(grads), flat_clipped
        ):
            mx.eval(g_orig, g_clip)
            np.testing.assert_allclose(
                np.array(g_orig), np.array(g_clip), atol=1e-6,
                err_msg="Gradients changed even though norm was within budget",
            )


# ---------------------------------------------------------------------------
# Property 20: Checkpoint Determinism
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for checkpoint round-trip testing."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(d, d)
        self.linear2 = nn.Linear(d, d)

    def __call__(self, x):
        return self.linear2(nn.relu(self.linear1(x)))


@given(
    d=st.sampled_from([8, 16, 32]),
    step=st.integers(min_value=0, max_value=1000),
)
@settings(deadline=None, max_examples=10)
def test_checkpoint_determinism(d, step):
    """**Validates: Requirements 7.4, 7.5**

    Property 20: Saving a checkpoint and loading it into a fresh model
    produces identical outputs for the same input.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create and initialise model + optimizer
        model = _TinyModel(d=d)
        optimizer = optim.Adam(learning_rate=1e-3)
        # Force evaluation of initial params
        mx.eval(model.parameters())

        # Fixed input for comparison
        x = mx.random.normal(shape=(2, d))
        mx.eval(x)

        # Forward pass before save
        out_before = model(x)
        mx.eval(out_before)
        out_before_np = np.array(out_before)

        # Save checkpoint
        mgr = CheckpointManager(checkpoint_dir=tmp_dir)
        ckpt_path = mgr.save(model, optimizer, step)

        # Create a fresh model with different random weights
        model2 = _TinyModel(d=d)
        optimizer2 = optim.Adam(learning_rate=1e-3)
        mx.eval(model2.parameters())

        # Load checkpoint into fresh model
        loaded_step = mgr.load(model2, optimizer2, ckpt_path)

        assert loaded_step == step, (
            f"Step mismatch: saved {step}, loaded {loaded_step}"
        )

        # Forward pass after load
        out_after = model2(x)
        mx.eval(out_after)
        out_after_np = np.array(out_after)

        np.testing.assert_allclose(
            out_before_np,
            out_after_np,
            atol=1e-5,
            err_msg=(
                f"Checkpoint round-trip produced different outputs. "
                f"Max diff: {np.max(np.abs(out_before_np - out_after_np))}"
            ),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
