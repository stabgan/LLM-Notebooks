"""Property-based tests for MoE router weight normalization.

**Validates: Requirement 1.2**
Router weights sum to 1.0 (±1e-6) along the last axis for each token,
all weights are non-negative, indices are valid, and shapes are correct.
"""

import numpy as np
import mlx.core as mx
from hypothesis import given, settings, strategies as st

from utils.moe import MoEConfig, MoERouter, ExpertChoiceRouter, HashRouter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keep ranges small to avoid slow tests / OOM on CI
batch_st = st.integers(min_value=1, max_value=3)
seq_st = st.integers(min_value=1, max_value=8)
d_model_st = st.sampled_from([16, 32, 64])
num_experts_st = st.integers(min_value=2, max_value=8)


@st.composite
def moe_config_st(draw):
    """Generate a valid MoEConfig with num_active <= num_experts."""
    d_model = draw(d_model_st)
    num_experts = draw(num_experts_st)
    num_active = draw(st.integers(min_value=1, max_value=min(num_experts, 4)))
    return MoEConfig(
        d_model=d_model,
        num_experts=num_experts,
        num_active=num_active,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_router_output(indices, weights, batch, seq, num_active, num_experts):
    """Shared assertions for all router types."""
    mx.eval(indices, weights)
    idx_np = np.array(indices)
    w_np = np.array(weights)

    # Shape checks
    assert idx_np.shape == (batch, seq, num_active), (
        f"indices shape {idx_np.shape} != expected {(batch, seq, num_active)}"
    )
    assert w_np.shape == (batch, seq, num_active), (
        f"weights shape {w_np.shape} != expected {(batch, seq, num_active)}"
    )

    # Weights sum to 1.0 (±1e-6) for every token
    weight_sums = w_np.sum(axis=-1)
    assert np.allclose(weight_sums, 1.0, atol=1e-6), (
        f"weight sums not ≈1.0: min={weight_sums.min()}, max={weight_sums.max()}"
    )

    # All weights non-negative
    assert np.all(w_np >= 0), f"negative weights found: min={w_np.min()}"

    # All indices in [0, num_experts)
    assert np.all(idx_np >= 0), f"negative indices found: min={idx_np.min()}"
    assert np.all(idx_np < num_experts), (
        f"indices out of range: max={idx_np.max()}, num_experts={num_experts}"
    )


# ---------------------------------------------------------------------------
# Property 3: Router Weight Normalization — MoERouter (Top-K)
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_config_st())
@settings(deadline=None, max_examples=30)
def test_topk_router_weight_normalization(batch, seq, config):
    """**Validates: Requirement 1.2**

    Property 3: For any random input, MoERouter (Top-K) routing weights
    sum to 1.0 (±1e-6), are non-negative, and indices are in [0, num_experts).
    """
    router = MoERouter(config)
    x = mx.random.normal((batch, seq, config.d_model))
    indices, weights = router.route(x)
    _check_router_output(
        indices, weights, batch, seq, config.num_active, config.num_experts
    )


# ---------------------------------------------------------------------------
# Property 3: Router Weight Normalization — ExpertChoiceRouter
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_config_st())
@settings(deadline=None, max_examples=30)
def test_expert_choice_router_weight_normalization(batch, seq, config):
    """**Validates: Requirement 1.2**

    Property 3: For any random input, ExpertChoiceRouter routing weights
    sum to 1.0 (±1e-6), are non-negative, and indices are in [0, num_experts).
    """
    router = ExpertChoiceRouter(config)
    x = mx.random.normal((batch, seq, config.d_model))
    indices, weights = router.route(x)
    _check_router_output(
        indices, weights, batch, seq, config.num_active, config.num_experts
    )


# ---------------------------------------------------------------------------
# Property 3: Router Weight Normalization — HashRouter
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_config_st())
@settings(deadline=None, max_examples=30)
def test_hash_router_weight_normalization(batch, seq, config):
    """**Validates: Requirement 1.2**

    Property 3: For any random input, HashRouter routing weights
    sum to 1.0 (±1e-6), are non-negative, and indices are in [0, num_experts).
    """
    router = HashRouter(config)
    x = mx.random.normal((batch, seq, config.d_model))
    indices, weights = router.route(x)
    _check_router_output(
        indices, weights, batch, seq, config.num_active, config.num_experts
    )


# ---------------------------------------------------------------------------
# Strategy for MoE block tests (smaller d_ff for speed)
# ---------------------------------------------------------------------------

@st.composite
def moe_block_config_st(draw):
    """Generate a valid MoEConfig suitable for MoEBlock forward-pass tests.

    Uses smaller d_ff to keep tests fast while still exercising the logic.
    """
    d_model = draw(d_model_st)
    num_experts = draw(st.integers(min_value=2, max_value=4))
    num_active = draw(st.integers(min_value=1, max_value=min(num_experts, 2)))
    d_ff = draw(st.sampled_from([32, 64]))
    return MoEConfig(
        d_model=d_model,
        num_experts=num_experts,
        num_active=num_active,
        d_ff=d_ff,
    )


# ---------------------------------------------------------------------------
# Property 1: Shape Preservation — MoE_Block
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_block_config_st())
@settings(deadline=None, max_examples=20)
def test_moe_block_shape_preservation(batch, seq, config):
    """**Validates: Requirements 1.4, 1.9**

    Property 1: For any random config and input of shape [batch, seq, d_model],
    MoEBlock output shape equals input shape [batch, seq, d_model].
    """
    from utils.moe import MoEBlock

    block = MoEBlock(config)
    x = mx.random.normal((batch, seq, config.d_model))
    output, aux_loss = block(x)
    mx.eval(output, aux_loss)

    out_np = np.array(output)
    assert out_np.shape == (batch, seq, config.d_model), (
        f"MoEBlock output shape {out_np.shape} != "
        f"expected {(batch, seq, config.d_model)}"
    )

    # Output should be free of NaN / Inf
    assert np.all(np.isfinite(out_np)), "MoEBlock output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Property 4: Expert Count Invariant
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_block_config_st())
@settings(deadline=None, max_examples=20)
def test_moe_block_expert_count_invariant(batch, seq, config):
    """**Validates: Requirements 1.4, 1.5**

    Property 4: For any input and MoE configuration, the router assigns
    each token to exactly num_active experts — i.e. the routing indices
    tensor has shape [batch, seq, num_active].
    """
    router = MoERouter(config)
    x = mx.random.normal((batch, seq, config.d_model))
    indices, weights = router.route(x)
    mx.eval(indices, weights)

    idx_np = np.array(indices)

    # Each token must have exactly num_active expert slots
    assert idx_np.shape == (batch, seq, config.num_active), (
        f"Router indices shape {idx_np.shape} != "
        f"expected {(batch, seq, config.num_active)}"
    )

    # All indices must be valid expert ids
    assert np.all(idx_np >= 0) and np.all(idx_np < config.num_experts), (
        f"Expert indices out of range [0, {config.num_experts}): "
        f"min={idx_np.min()}, max={idx_np.max()}"
    )


# ---------------------------------------------------------------------------
# Property 5: Load Balance Loss Formula
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, config=moe_block_config_st())
@settings(deadline=None, max_examples=20)
def test_moe_block_load_balance_loss_formula(batch, seq, config):
    """**Validates: Requirements 1.5, 1.9**

    Property 5: The load balancing auxiliary loss equals
    num_experts × Σ(f_i × p_i) and is non-negative.
    """
    router = MoERouter(config)
    x = mx.random.normal((batch, seq, config.d_model))

    # Run routing to populate _last_logits
    indices, weights = router.route(x)
    mx.eval(indices, weights)

    # Compute loss via the implementation
    impl_loss = router.compute_load_balance_loss()
    mx.eval(impl_loss)
    impl_loss_val = float(impl_loss)

    # --- Manual recomputation of the formula ---
    logits = router._last_logits                       # [batch, seq, E]
    logits_flat = np.array(logits.reshape(-1, config.num_experts))  # [N, E]
    N = logits_flat.shape[0]

    # Recompute top-k indices (same logic as implementation)
    from utils.moe import top_k as mlx_top_k
    idx_flat = np.array(
        mlx_top_k(mx.array(logits_flat), config.num_active)[1]
    )  # [N, k]

    # f_i: fraction of tokens routed to each expert
    one_hot = np.zeros((N, config.num_experts), dtype=np.float64)
    for k_idx in range(config.num_active):
        for n in range(N):
            one_hot[n, idx_flat[n, k_idx]] += 1.0
    f = one_hot.mean(axis=0)  # [E]

    # p_i: mean routing probability (softmax over all experts)
    # Numerically stable softmax
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_flat - logits_max)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    p = probs.mean(axis=0)  # [E]

    expected_loss = float(config.num_experts * np.sum(f * p))

    # Non-negativity
    assert impl_loss_val >= -1e-7, (
        f"Load balance loss is negative: {impl_loss_val}"
    )

    # Formula match (allow small floating-point tolerance)
    assert abs(impl_loss_val - expected_loss) < 1e-3, (
        f"Load balance loss mismatch: impl={impl_loss_val}, "
        f"expected={expected_loss}, diff={abs(impl_loss_val - expected_loss)}"
    )
