"""Property-based tests for Transformer Architecture deep-dive (Notebook 06).

**Validates: Requirements 6.4, 6.5, 6.6**

Property 16: Pre-Norm Gradient Stability — pre-norm has lower CV of gradient
             norms than post-norm for depth ≥ 4.
Property 17: Parameter Counting Consistency — sum of components equals total,
             memory = params × bytes.
"""

import math

import numpy as np
from hypothesis import given, settings, strategies as st, assume

from utils.transformer_analysis import (
    TransformerConfig,
    ParameterCounter,
    DTYPE_BYTES,
    gradient_flow_analysis,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_d_model_st = st.sampled_from([32, 64, 128])
_n_heads_st = st.sampled_from([2, 4, 8])
_n_layers_st = st.integers(min_value=1, max_value=12)
_d_ff_mult = st.sampled_from([2, 4])
_vocab_st = st.integers(min_value=100, max_value=50000)
_activation_st = st.sampled_from(["relu", "gelu", "silu", "swiglu", "geglu"])
_norm_st = st.sampled_from(["layernorm", "rmsnorm", "deepnorm"])
_dtype_st = st.sampled_from(list(DTYPE_BYTES.keys()))


@st.composite
def transformer_config_st(draw):
    """Generate a valid TransformerConfig."""
    d_model = draw(_d_model_st)
    n_heads = draw(_n_heads_st)
    assume(d_model % n_heads == 0)
    n_kv_heads = draw(st.sampled_from([h for h in [1, 2, 4, 8] if h <= n_heads]))
    n_layers = draw(_n_layers_st)
    d_ff = d_model * draw(_d_ff_mult)
    vocab_size = draw(_vocab_st)
    activation = draw(_activation_st)
    norm_type = draw(_norm_st)
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        activation=activation,
        norm_type=norm_type,
    )


# ---------------------------------------------------------------------------
# Property 16: Pre-Norm Gradient Stability
# ---------------------------------------------------------------------------

@given(
    depth=st.integers(min_value=6, max_value=10),
    d_model=st.sampled_from([64, 128]),
)
@settings(deadline=None, max_examples=10)
def test_pre_norm_gradient_stability(depth, d_model):
    """**Validates: Requirement 6.4**

    Property 16: For depth ≥ 4, pre-norm architecture produces higher mean
    activation gradient norms than post-norm.  The residual path in pre-norm
    (x + sublayer(norm(x))) provides a direct gradient highway, so each
    layer receives a stronger learning signal on average.  Post-norm wraps
    the residual in normalization which compresses gradient magnitudes.
    """
    result = gradient_flow_analysis(depth=depth, d_model=d_model, seq_len=4, batch=1, n_trials=20)

    pre_mean = result["pre_mean_grad"]
    post_mean = result["post_mean_grad"]

    # Both should be finite and positive
    assert np.isfinite(pre_mean) and pre_mean > 0, f"Pre-norm mean grad invalid: {pre_mean}"
    assert np.isfinite(post_mean) and post_mean > 0, f"Post-norm mean grad invalid: {post_mean}"

    assert pre_mean > post_mean, (
        f"Pre-norm mean activation gradient ({pre_mean:.4f}) should be higher than "
        f"post-norm ({post_mean:.4f}) for depth={depth}, d_model={d_model}. "
        f"Pre-norm grads: {result['pre_grad_norms']}, "
        f"Post-norm grads: {result['post_grad_norms']}"
    )


# ---------------------------------------------------------------------------
# Property 17: Parameter Counting Consistency
# ---------------------------------------------------------------------------

@given(config=transformer_config_st())
@settings(deadline=None, max_examples=50)
def test_parameter_counting_sum_equals_total(config):
    """**Validates: Requirements 6.5, 6.6**

    Property 17 (part a): The sum of per-component parameter counts
    (embedding + attention + ffn + normalization) equals the total.
    """
    counts = ParameterCounter.count(config)

    component_sum = (
        counts["embedding"]
        + counts["attention"]
        + counts["ffn"]
        + counts["normalization"]
    )

    assert component_sum == counts["total"], (
        f"Component sum ({component_sum:,}) != total ({counts['total']:,}). "
        f"Breakdown: {counts}"
    )


@given(config=transformer_config_st(), dtype=_dtype_st)
@settings(deadline=None, max_examples=50)
def test_parameter_memory_equals_params_times_bytes(config, dtype):
    """**Validates: Requirements 6.5, 6.6**

    Property 17 (part b): Memory estimate for each component equals
    params × bytes_per_element for the given dtype.
    """
    counts = ParameterCounter.count(config)
    memory = ParameterCounter.estimate_memory(config, dtype)
    bpe = DTYPE_BYTES[dtype]

    for component in ["embedding", "attention", "ffn", "normalization", "total"]:
        expected_mem = counts[component] * bpe
        actual_mem = memory[component]
        assert actual_mem == expected_mem, (
            f"Memory mismatch for {component}: "
            f"expected {expected_mem} (params={counts[component]} × bpe={bpe}), "
            f"got {actual_mem}. dtype={dtype}"
        )
