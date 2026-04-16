"""Property-based tests for attention optimization utilities (Notebook 12).

**Validates: Requirements 10.2, 10.4**

Property 26: Online Softmax Equivalence — matches standard softmax within 1e-5.
Property 27: Flash Attention Equivalence — matches standard attention within 1e-5.
"""

import math

import mlx.core as mx
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from utils.attention_optimization import (
    online_softmax,
    online_softmax_blocked,
    standard_attention,
    tiled_attention,
)


# ---------------------------------------------------------------------------
# Helpers: Hypothesis strategies for MLX tensors
# ---------------------------------------------------------------------------

def _mlx_vector(size: int, seed: int, low: float = -10.0, high: float = 10.0) -> mx.array:
    """Generate a deterministic MLX vector from a seed."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(low, high, size=size).astype(np.float32)
    return mx.array(data)


def _mlx_matrix(rows: int, cols: int, seed: int,
                low: float = -5.0, high: float = 5.0) -> mx.array:
    """Generate a deterministic MLX matrix from a seed."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(low, high, size=(rows, cols)).astype(np.float32)
    return mx.array(data)


# ---------------------------------------------------------------------------
# Property 26: Online Softmax Equivalence
# ---------------------------------------------------------------------------

@given(
    rows=st.integers(min_value=1, max_value=8),
    cols=st.integers(min_value=2, max_value=64),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=50)
def test_online_softmax_equivalence(rows, cols, seed):
    """**Validates: Requirement 10.2**

    Property 26: Online Softmax Equivalence — for any attention score matrix
    partitioned into blocks, the online softmax algorithm produces results
    identical to standard two-pass softmax within floating-point tolerance of 1e-5.

    We generate random input tensors and verify that online_softmax matches
    mx.softmax element-wise within 1e-5.
    """
    x = _mlx_matrix(rows, cols, seed, low=-10.0, high=10.0)
    mx.eval(x)

    # Standard softmax (reference)
    expected = mx.softmax(x, axis=-1)

    # Online softmax (element-by-element streaming)
    result = online_softmax(x)

    mx.eval(expected, result)

    diff = float(mx.max(mx.abs(expected - result)))
    assert diff < 1e-5, (
        f"Online softmax diverged from standard softmax: max_diff={diff:.2e} "
        f"(rows={rows}, cols={cols}, seed={seed})"
    )

    # Also verify basic softmax properties: non-negative, sums to 1
    result_np = np.array(result)
    assert np.all(result_np >= 0), "Softmax output contains negative values"
    row_sums = result_np.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"Softmax rows don't sum to 1: {row_sums}"
    )


@given(
    rows=st.integers(min_value=1, max_value=8),
    cols=st.integers(min_value=2, max_value=64),
    block_size=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=50)
def test_online_softmax_blocked_equivalence(rows, cols, block_size, seed):
    """**Validates: Requirement 10.2**

    Property 26 (blocked variant): The blocked online softmax also matches
    standard softmax within 1e-5, regardless of block size.
    """
    x = _mlx_matrix(rows, cols, seed, low=-10.0, high=10.0)
    mx.eval(x)

    expected = mx.softmax(x, axis=-1)
    result = online_softmax_blocked(x, block_size=block_size)
    mx.eval(expected, result)

    diff = float(mx.max(mx.abs(expected - result)))
    assert diff < 1e-5, (
        f"Blocked online softmax diverged: max_diff={diff:.2e} "
        f"(rows={rows}, cols={cols}, block_size={block_size}, seed={seed})"
    )


# ---------------------------------------------------------------------------
# Property 27: Flash Attention Equivalence
# ---------------------------------------------------------------------------

@given(
    seq_len=st.integers(min_value=4, max_value=64),
    d=st.sampled_from([8, 16, 32]),
    block_size=st.sampled_from([2, 4, 8, 16]),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=50)
def test_flash_attention_equivalence(seq_len, d, block_size, seed):
    """**Validates: Requirement 10.4**

    Property 27: Flash Attention Equivalence — for any query, key, and value
    tensors, TiledFlashAttention(Q, K, V) produces output identical to
    standard attention softmax(QK^T/√d)V within tolerance of 1e-5.

    We generate random Q, K, V matrices and compare tiled flash attention
    output against the standard O(n²) reference implementation.
    """
    Q = _mlx_matrix(seq_len, d, seed, low=-3.0, high=3.0)
    K = _mlx_matrix(seq_len, d, seed + 1, low=-3.0, high=3.0)
    V = _mlx_matrix(seq_len, d, seed + 2, low=-3.0, high=3.0)
    mx.eval(Q, K, V)

    # Standard attention (reference)
    expected = standard_attention(Q, K, V)

    # Tiled flash attention
    result = tiled_attention(Q, K, V, block_size=block_size)

    mx.eval(expected, result)

    # Shape must match
    assert expected.shape == result.shape, (
        f"Shape mismatch: expected {expected.shape}, got {result.shape}"
    )

    diff = float(mx.max(mx.abs(expected - result)))
    assert diff < 1e-5, (
        f"Flash attention diverged from standard: max_diff={diff:.2e} "
        f"(seq_len={seq_len}, d={d}, block_size={block_size}, seed={seed})"
    )
