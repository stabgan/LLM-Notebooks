"""Property-based tests for Scaling Laws (Notebook 18).

**Validates: Requirements 4.1, 4.2, 4.3**

Property 13: Scaling Law Monotonicity — loss decreases as N or D increases.
Property 14: Compute Budget Conservation — |6 × N × D - C| / C ≤ 0.10.
Property 28: Scaling Law Formula — L(N,D) = A/N^α + B/D^β + E.
"""

import numpy as np
from hypothesis import given, settings, strategies as st

from utils.scaling import ScalingLawParams, ScalingLawPredictor


# ---------------------------------------------------------------------------
# Strategies — constrain to realistic positive ranges
# ---------------------------------------------------------------------------

# Model size N (parameters): 1e6 to 1e12
n_st = st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False)

# Data size D (tokens): 1e6 to 1e12
d_st = st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False)

# Compute budget C (FLOPs): 1e15 to 1e25
c_st = st.floats(min_value=1e15, max_value=1e25, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 13: Scaling Law Monotonicity
# ---------------------------------------------------------------------------

@given(
    n1=st.floats(min_value=1e6, max_value=1e11, allow_nan=False, allow_infinity=False),
    n2_delta=st.floats(min_value=1e6, max_value=1e11, allow_nan=False, allow_infinity=False),
    d=d_st,
)
@settings(deadline=None, max_examples=50)
def test_scaling_law_monotonicity_N(n1, n2_delta, d):
    """**Validates: Requirement 4.2**

    Property 13 (part a): For any fixed data size D and two model sizes
    N1 < N2, the predicted loss L(N1, D) > L(N2, D).

    Larger models should always achieve lower loss.
    """
    n2 = n1 + n2_delta  # guarantees n2 > n1
    predictor = ScalingLawPredictor()

    loss_n1 = predictor.predict_loss(n1, d)
    loss_n2 = predictor.predict_loss(n2, d)

    assert loss_n1 > loss_n2, (
        f"Monotonicity violated for N: L(N1={n1:.2e}, D={d:.2e})={loss_n1:.6f} "
        f"should be > L(N2={n2:.2e}, D={d:.2e})={loss_n2:.6f}"
    )


@given(
    n=n_st,
    d1=st.floats(min_value=1e6, max_value=1e11, allow_nan=False, allow_infinity=False),
    d2_delta=st.floats(min_value=1e6, max_value=1e11, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=50)
def test_scaling_law_monotonicity_D(n, d1, d2_delta):
    """**Validates: Requirement 4.2**

    Property 13 (part b): For any fixed model size N and two data sizes
    D1 < D2, the predicted loss L(N, D1) > L(N, D2).

    More training data should always reduce loss.
    """
    d2 = d1 + d2_delta  # guarantees d2 > d1
    predictor = ScalingLawPredictor()

    loss_d1 = predictor.predict_loss(n, d1)
    loss_d2 = predictor.predict_loss(n, d2)

    assert loss_d1 > loss_d2, (
        f"Monotonicity violated for D: L(N={n:.2e}, D1={d1:.2e})={loss_d1:.6f} "
        f"should be > L(N={n:.2e}, D2={d2:.2e})={loss_d2:.6f}"
    )


# ---------------------------------------------------------------------------
# Property 14: Compute Budget Conservation
# ---------------------------------------------------------------------------

@given(c=c_st)
@settings(deadline=None, max_examples=50)
def test_compute_budget_conservation(c):
    """**Validates: Requirement 4.3**

    Property 14: For any positive compute budget C, the optimal allocation
    satisfies |6 × optimal_N × optimal_D - C| / C ≤ 0.10.
    """
    predictor = ScalingLawPredictor()
    budget = predictor.compute_optimal_allocation(c)

    actual_flops = 6.0 * budget.optimal_N * budget.optimal_D
    relative_error = abs(actual_flops - c) / c

    assert relative_error <= 0.10, (
        f"Budget conservation violated: C={c:.2e}, "
        f"6*N*D={actual_flops:.2e}, relative_error={relative_error:.6f} > 0.10"
    )


# ---------------------------------------------------------------------------
# Property 28: Scaling Law Formula
# ---------------------------------------------------------------------------

@given(n=n_st, d=d_st)
@settings(deadline=None, max_examples=50)
def test_scaling_law_formula(n, d):
    """**Validates: Requirement 4.1**

    Property 28: For any model size N and data size D, predict_loss(N, D)
    SHALL equal A/N^α + B/D^β + E using the calibrated Chinchilla constants.
    """
    params = ScalingLawParams()
    predictor = ScalingLawPredictor(params)

    impl_loss = predictor.predict_loss(n, d)

    # Manual computation of the Chinchilla formula
    expected_loss = (
        params.A / (n ** params.alpha)
        + params.B / (d ** params.beta)
        + params.E
    )

    assert np.isfinite(impl_loss), f"predict_loss returned non-finite: {impl_loss}"
    assert np.isfinite(expected_loss), f"Manual formula non-finite: {expected_loss}"

    assert abs(impl_loss - expected_loss) < 1e-6, (
        f"Formula mismatch: predict_loss({n:.2e}, {d:.2e})={impl_loss:.8f}, "
        f"expected A/N^α + B/D^β + E = {expected_loss:.8f}, "
        f"diff={abs(impl_loss - expected_loss):.2e}"
    )
