"""Property-based tests for SSM discretization and causality.

**Validates: Requirements 2.2, 2.4**

Property 6: SSM Discretization Correctness — A_bar = exp(delta × A),
    B_bar = delta × B for valid inputs, and |A_bar| <= 1.0 when A < 0, delta > 0.

Property 7: SSM Causality — output at position t depends only on inputs 0..t.
"""

import numpy as np
import mlx.core as mx
from hypothesis import given, settings, strategies as st

from utils.ssm import SSMConfig, SimpleSSM


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

d_inner_st = st.sampled_from([8, 16, 32])
d_state_st = st.sampled_from([4, 8, 16])


@st.composite
def ssm_config_st(draw):
    """Generate a valid SSMConfig with small dimensions for fast tests."""
    d_model = draw(st.sampled_from([16, 32]))
    d_state = draw(d_state_st)
    d_inner = draw(d_inner_st)
    return SSMConfig(d_model=d_model, d_state=d_state, d_inner=d_inner)


# ---------------------------------------------------------------------------
# Property 6: SSM Discretization Correctness
# ---------------------------------------------------------------------------

@given(
    d_inner=d_inner_st,
    d_state=d_state_st,
    delta_scale=st.floats(min_value=0.01, max_value=2.0),
)
@settings(deadline=None, max_examples=50)
def test_ssm_discretization_correctness(d_inner, d_state, delta_scale):
    """**Validates: Requirements 2.2**

    Property 6: For random A (negative diagonal), B, and positive delta,
    verify:
      - A_bar = exp(delta * A) element-wise (within 1e-5)
      - B_bar = delta * B element-wise (within 1e-5)
      - |A_bar| <= 1.0 when A is negative and delta > 0
    """
    # Build inputs: A with negative values, B random, delta positive
    A = -mx.abs(mx.random.normal((d_inner, d_state))) - 0.1  # strictly negative
    B = mx.random.normal((d_inner, d_state)) * 0.01
    delta = mx.abs(mx.random.normal((d_inner,))) * delta_scale + 1e-4  # positive

    # Run discretization
    A_bar, B_bar = SimpleSSM.discretize(A, B, delta)
    mx.eval(A_bar, B_bar)

    # --- Manual reference computation ---
    A_np = np.array(A)
    B_np = np.array(B)
    delta_np = np.array(delta)

    # A_bar = exp(delta * A)
    expected_A_bar = np.exp(delta_np[:, None] * A_np)
    # B_bar = delta * B
    expected_B_bar = delta_np[:, None] * B_np

    A_bar_np = np.array(A_bar)
    B_bar_np = np.array(B_bar)

    # Check A_bar matches within tolerance
    assert np.allclose(A_bar_np, expected_A_bar, atol=1e-5), (
        f"A_bar mismatch: max diff = {np.max(np.abs(A_bar_np - expected_A_bar))}"
    )

    # Check B_bar matches within tolerance
    assert np.allclose(B_bar_np, expected_B_bar, atol=1e-5), (
        f"B_bar mismatch: max diff = {np.max(np.abs(B_bar_np - expected_B_bar))}"
    )

    # Stability: |A_bar| <= 1.0 when A < 0 and delta > 0
    # exp(delta * A) where delta > 0 and A < 0 => exponent < 0 => result in (0, 1)
    assert np.all(np.abs(A_bar_np) <= 1.0 + 1e-7), (
        f"|A_bar| exceeds 1.0: max = {np.max(np.abs(A_bar_np))}"
    )

    # Shape checks
    assert A_bar_np.shape == (d_inner, d_state)
    assert B_bar_np.shape == (d_inner, d_state)


# ---------------------------------------------------------------------------
# Property 7: SSM Causality
# ---------------------------------------------------------------------------

@given(
    config=ssm_config_st(),
    batch=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=4, max_value=16),
)
@settings(deadline=None, max_examples=30)
def test_ssm_causality(config, batch, seq_len):
    """**Validates: Requirements 2.4**

    Property 7: For random inputs, verify:
      - Modifying input at position t+k does NOT change output at positions 0..t
      - Modifying input at position t DOES change output at position t
    """
    d_inner = config.d_inner
    ssm = SimpleSSM(config)

    # Original input
    x_orig = mx.random.normal((batch, seq_len, d_inner))
    y_orig = ssm(x_orig)
    mx.eval(y_orig)
    y_orig_np = np.array(y_orig)

    # Pick a split point t in [1, seq_len - 2] so we have room to perturb after t
    t = seq_len // 2

    # --- Perturb input AFTER position t ---
    perturbation = mx.random.normal((batch, seq_len - t - 1, d_inner)) * 10.0
    x_perturbed = mx.array(np.array(x_orig))  # copy
    # Replace positions t+1 onwards
    x_perturbed_np = np.array(x_perturbed)
    x_perturbed_np[:, t + 1 :, :] = np.array(perturbation)
    x_perturbed = mx.array(x_perturbed_np)

    y_perturbed = ssm(x_perturbed)
    mx.eval(y_perturbed)
    y_perturbed_np = np.array(y_perturbed)

    # Outputs at positions 0..t should be UNCHANGED
    assert np.allclose(y_orig_np[:, : t + 1, :], y_perturbed_np[:, : t + 1, :], atol=1e-5), (
        f"Causality violated: output at positions 0..{t} changed when input "
        f"after position {t} was modified. "
        f"Max diff = {np.max(np.abs(y_orig_np[:, :t+1, :] - y_perturbed_np[:, :t+1, :]))}"
    )

    # --- Perturb input AT position t ---
    x_at_t = mx.array(np.array(x_orig))  # fresh copy
    x_at_t_np = np.array(x_at_t)
    x_at_t_np[:, t, :] += 10.0  # large perturbation
    x_at_t = mx.array(x_at_t_np)

    y_at_t = ssm(x_at_t)
    mx.eval(y_at_t)
    y_at_t_np = np.array(y_at_t)

    # Output at position t SHOULD change
    diff_at_t = np.max(np.abs(y_orig_np[:, t, :] - y_at_t_np[:, t, :]))
    assert diff_at_t > 1e-6, (
        f"Causality check: output at position {t} did NOT change when input "
        f"at position {t} was modified by +10.0. Max diff = {diff_at_t}"
    )

    # Outputs at positions 0..t-1 should still be UNCHANGED
    if t > 0:
        assert np.allclose(y_orig_np[:, :t, :], y_at_t_np[:, :t, :], atol=1e-5), (
            f"Causality violated: output at positions 0..{t-1} changed when "
            f"only input at position {t} was modified. "
            f"Max diff = {np.max(np.abs(y_orig_np[:, :t, :] - y_at_t_np[:, :t, :]))}"
        )


# === Additional imports for Properties 8 & 9 ===
from utils.ssm import SelectiveSSM


# ---------------------------------------------------------------------------
# Strategies for Properties 8 & 9
# ---------------------------------------------------------------------------

# Mix of power-of-2 and non-power-of-2 sequence lengths
seq_len_st = st.sampled_from([2, 3, 4, 5, 7, 8, 10, 13, 16])


# ---------------------------------------------------------------------------
# Property 8: Sequential vs Parallel Scan Equivalence
# ---------------------------------------------------------------------------

@given(
    config=ssm_config_st(),
    batch=st.integers(min_value=1, max_value=2),
    seq_len=seq_len_st,
)
@settings(deadline=None, max_examples=50)
def test_sequential_vs_parallel_scan_equivalence(config, batch, seq_len):
    """**Validates: Requirements 2.5**

    Property 8: For any valid input sequence, the sequential scan and
    parallel scan implementations SHALL produce identical outputs within
    floating-point tolerance of 1e-5.

    Tests with both power-of-2 (2, 4, 8, 16) and non-power-of-2
    (3, 5, 7, 10, 13) sequence lengths.
    """
    d_inner = config.d_inner

    # Build a SelectiveSSM — weights are fixed after construction
    model = SelectiveSSM(config)

    # Generate a random input
    x = mx.random.normal((batch, seq_len, d_inner)) * 0.1
    mx.eval(x)

    # Run through sequential scan
    y_seq = model(x, use_parallel=False)
    mx.eval(y_seq)

    # Run through parallel scan (same model, same input)
    y_par = model(x, use_parallel=True)
    mx.eval(y_par)

    y_seq_np = np.array(y_seq)
    y_par_np = np.array(y_par)

    # Shape check
    assert y_seq_np.shape == (batch, seq_len, d_inner), (
        f"Sequential output shape {y_seq_np.shape} != ({batch}, {seq_len}, {d_inner})"
    )
    assert y_par_np.shape == (batch, seq_len, d_inner), (
        f"Parallel output shape {y_par_np.shape} != ({batch}, {seq_len}, {d_inner})"
    )

    # Equivalence within 1e-5
    max_diff = np.max(np.abs(y_seq_np - y_par_np))
    assert np.allclose(y_seq_np, y_par_np, atol=1e-5), (
        f"Sequential vs parallel scan mismatch: max diff = {max_diff:.2e} "
        f"(config: d_model={config.d_model}, d_inner={d_inner}, "
        f"d_state={config.d_state}, batch={batch}, seq_len={seq_len})"
    )

    # Both outputs should be finite (no NaN/Inf)
    assert np.all(np.isfinite(y_seq_np)), "Sequential scan produced NaN/Inf"
    assert np.all(np.isfinite(y_par_np)), "Parallel scan produced NaN/Inf"


# ---------------------------------------------------------------------------
# Property 9: SSM State Boundedness
# ---------------------------------------------------------------------------

@given(
    config=ssm_config_st(),
    batch=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=64, max_value=256),
    input_scale=st.floats(min_value=0.01, max_value=1.0),
)
@settings(deadline=None, max_examples=30)
def test_ssm_state_boundedness(config, batch, seq_len, input_scale):
    """**Validates: Requirements 2.6**

    Property 9: For a stable A matrix (all negative values, so
    |A_bar| = |exp(delta * A)| < 1) and bounded input, the hidden
    state norm SHALL remain bounded as sequence length increases.

    Checks:
      - Output contains no NaN or Inf values
      - Output norm does not explode (stays within a reasonable bound)
      - Output at the end of a long sequence is not larger than at the start
        (no unbounded growth)
    """
    d_inner = config.d_inner

    # SimpleSSM initialises A with negative values (-1, -2, ..., -N),
    # guaranteeing |A_bar| < 1 for positive delta → stable recurrence.
    model = SimpleSSM(config)

    # Bounded input: values in [-input_scale, input_scale]
    x = mx.random.normal((batch, seq_len, d_inner)) * input_scale
    mx.eval(x)

    y = model(x)
    mx.eval(y)
    y_np = np.array(y)

    # --- Check 1: No NaN or Inf ---
    assert np.all(np.isfinite(y_np)), (
        f"SSM output contains NaN/Inf for seq_len={seq_len}, "
        f"input_scale={input_scale}"
    )

    # --- Check 2: Output is bounded ---
    # For a stable SSM with |A_bar| < 1, the steady-state output norm
    # is bounded by roughly |B_bar * x_max| / (1 - |A_bar_max|).
    # We use a generous bound: output should not exceed 1000× input scale.
    max_output = np.max(np.abs(y_np))
    bound = 1000.0 * max(input_scale, 1e-3)
    assert max_output < bound, (
        f"SSM output appears unbounded: max |y| = {max_output:.4f}, "
        f"expected < {bound:.4f} for input_scale={input_scale}"
    )

    # --- Check 3: No unbounded growth over time ---
    # Compare norm of output in the last quarter vs first quarter.
    # For a stable system, the later output should not be dramatically
    # larger than the earlier output (no exponential blowup).
    quarter = max(seq_len // 4, 1)
    norm_first = np.mean(np.abs(y_np[:, :quarter, :]))
    norm_last = np.mean(np.abs(y_np[:, -quarter:, :]))

    # Allow the last quarter to be up to 10× the first (transient effects),
    # but not orders of magnitude larger (which would indicate instability).
    if norm_first > 1e-8:
        growth_ratio = norm_last / norm_first
        assert growth_ratio < 100.0, (
            f"SSM output growing over time: last/first norm ratio = "
            f"{growth_ratio:.2f} (seq_len={seq_len}). "
            f"This suggests unstable dynamics."
        )
