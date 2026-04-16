"""Property-based tests for DPO alignment (Notebook 17).

**Validates: Requirements 3.3, 3.4, 3.5**

Property 10: DPO Loss Correctness — loss matches formula and is non-negative.
Property 11: Reference Model Frozen — reference params unchanged after training steps.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from hypothesis import given, settings, strategies as st

from utils.alignment import DPOConfig, DPOTrainer, SimpleLM, RewardModelConfig


# ---------------------------------------------------------------------------
# Fixed small model config (fast tests, low memory)
# ---------------------------------------------------------------------------

SMALL_CONFIG = RewardModelConfig(
    d_model=32,
    n_heads=4,
    n_layers=1,
    vocab_size=64,
    max_seq_len=16,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

batch_st = st.integers(min_value=1, max_value=3)
seq_st = st.integers(min_value=3, max_value=12)  # ≥2 for next-token pred
beta_st = st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trainer(beta: float) -> DPOTrainer:
    """Create a DPOTrainer with fresh small policy and reference models."""
    policy = SimpleLM(SMALL_CONFIG)
    reference = SimpleLM(SMALL_CONFIG)
    mx.eval(policy.parameters(), reference.parameters())
    dpo_cfg = DPOConfig(beta=beta, learning_rate=1e-4, max_length=SMALL_CONFIG.max_seq_len)
    return DPOTrainer(dpo_cfg, policy, reference)


def _random_ids(batch: int, seq: int) -> mx.array:
    """Generate random token IDs in [0, vocab_size)."""
    return mx.random.randint(0, SMALL_CONFIG.vocab_size, (batch, seq))


def _flatten_params(model: nn.Module) -> dict[str, np.ndarray]:
    """Snapshot all model parameters as numpy arrays keyed by path."""
    leaves = model.parameters()
    flat: dict[str, np.ndarray] = {}

    def _walk(obj, prefix=""):
        if isinstance(obj, mx.array):
            mx.eval(obj)
            flat[prefix] = np.array(obj, copy=True)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{prefix}[{i}]")

    _walk(leaves)
    return flat


# ---------------------------------------------------------------------------
# Property 10: DPO Loss Non-Negative
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, beta=beta_st)
@settings(deadline=None, max_examples=30)
def test_dpo_loss_non_negative(batch, seq, beta):
    """**Validates: Requirements 3.3, 3.4**

    Property 10 (part a): For any random preference pair and positive beta,
    the DPO loss is non-negative and finite (no NaN/Inf).

    Mathematical basis: DPO loss = -log(σ(x)) = softplus(-x) ≥ 0.
    """
    trainer = _make_trainer(beta)
    chosen_ids = _random_ids(batch, seq)
    rejected_ids = _random_ids(batch, seq)

    loss = trainer.dpo_loss(chosen_ids, rejected_ids)
    mx.eval(loss)
    loss_val = float(loss)

    # Loss must be finite
    assert np.isfinite(loss_val), f"DPO loss is not finite: {loss_val}"

    # Loss must be non-negative (softplus is always ≥ 0)
    assert loss_val >= -1e-7, f"DPO loss is negative: {loss_val}"


# ---------------------------------------------------------------------------
# Property 10: DPO Loss Formula Correctness
# ---------------------------------------------------------------------------

@given(batch=batch_st, seq=seq_st, beta=beta_st)
@settings(deadline=None, max_examples=30)
def test_dpo_loss_formula(batch, seq, beta):
    """**Validates: Requirements 3.3, 3.4**

    Property 10 (part b): The DPO loss matches the manual formula:
    loss = -mean(log σ(β × (log_ratio_chosen - log_ratio_rejected)))
    where log_ratio = log π_policy - log π_reference.
    """
    trainer = _make_trainer(beta)
    chosen_ids = _random_ids(batch, seq)
    rejected_ids = _random_ids(batch, seq)

    # --- Implementation result ---
    impl_loss = trainer.dpo_loss(chosen_ids, rejected_ids)
    mx.eval(impl_loss)
    impl_val = float(impl_loss)

    # --- Manual recomputation ---
    # Log-probs under policy
    log_pi_chosen = DPOTrainer.compute_log_probs(trainer.policy_model, chosen_ids)
    log_pi_rejected = DPOTrainer.compute_log_probs(trainer.policy_model, rejected_ids)

    # Log-probs under reference (frozen)
    log_ref_chosen = DPOTrainer.compute_log_probs(trainer.reference_model, chosen_ids)
    log_ref_rejected = DPOTrainer.compute_log_probs(trainer.reference_model, rejected_ids)

    mx.eval(log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected)

    # Convert to numpy for manual computation
    lpc = float(log_pi_chosen) if batch == 1 else np.array(log_pi_chosen)
    lpr = float(log_pi_rejected) if batch == 1 else np.array(log_pi_rejected)
    lrc = float(log_ref_chosen) if batch == 1 else np.array(log_ref_chosen)
    lrr = float(log_ref_rejected) if batch == 1 else np.array(log_ref_rejected)

    # Log-ratios
    log_ratio_chosen = np.array(lpc) - np.array(lrc)
    log_ratio_rejected = np.array(lpr) - np.array(lrr)

    # DPO logits
    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    # Numerically stable log-sigmoid: log σ(x) = x - softplus(x) = -softplus(-x)
    log_sigmoid = -np.logaddexp(0.0, -logits)
    expected_loss = float(-np.mean(log_sigmoid))

    # Compare
    assert np.isfinite(expected_loss), f"Manual loss is not finite: {expected_loss}"
    assert abs(impl_val - expected_loss) < 1e-3, (
        f"DPO loss mismatch: impl={impl_val}, expected={expected_loss}, "
        f"diff={abs(impl_val - expected_loss)}"
    )


# ---------------------------------------------------------------------------
# Property 11: Reference Model Frozen
# ---------------------------------------------------------------------------

@given(
    batch=batch_st,
    seq=seq_st,
    num_steps=st.integers(min_value=1, max_value=3),
)
@settings(deadline=None, max_examples=15)
def test_reference_model_frozen(batch, seq, num_steps):
    """**Validates: Requirement 3.5**

    Property 11: After any number of DPO training steps, the reference
    model parameters remain identical to their initial values.
    """
    trainer = _make_trainer(beta=0.1)

    # Snapshot reference params before training
    ref_before = _flatten_params(trainer.reference_model)
    assert len(ref_before) > 0, "Reference model has no parameters"

    # Run several training steps
    for _ in range(num_steps):
        chosen_ids = _random_ids(batch, seq)
        rejected_ids = _random_ids(batch, seq)
        trainer.train_step(chosen_ids, rejected_ids)

    # Snapshot reference params after training
    ref_after = _flatten_params(trainer.reference_model)

    # Every parameter must be identical
    assert set(ref_before.keys()) == set(ref_after.keys()), (
        "Reference model parameter keys changed after training"
    )
    for key in ref_before:
        np.testing.assert_array_equal(
            ref_before[key],
            ref_after[key],
            err_msg=f"Reference param '{key}' changed after {num_steps} train steps",
        )


# ---------------------------------------------------------------------------
# Import GRPOTrainer for Property 12
# ---------------------------------------------------------------------------

from utils.alignment import GRPOTrainer


# ---------------------------------------------------------------------------
# Strategies for GRPO reward normalization
# ---------------------------------------------------------------------------

reward_size_st = st.integers(min_value=2, max_value=50)


# ---------------------------------------------------------------------------
# Property 12: GRPO Reward Normalization
# ---------------------------------------------------------------------------

@given(
    n=reward_size_st,
    data=st.data(),
)
@settings(deadline=None, max_examples=50)
def test_grpo_reward_normalization(n, data):
    """**Validates: Requirement 3.6**

    Property 12: For any group of responses with non-zero reward variance,
    the normalized rewards SHALL have mean ≈ 0 and std ≈ 1.

    Tests with various reward distributions: uniform, skewed, and
    near-constant (but with non-zero variance).
    """
    # Draw a distribution type to get variety
    dist_type = data.draw(st.sampled_from(["uniform", "skewed", "spread"]))

    if dist_type == "uniform":
        # Uniform rewards in a random range
        low = data.draw(st.floats(min_value=-100.0, max_value=100.0,
                                  allow_nan=False, allow_infinity=False))
        high = low + data.draw(st.floats(min_value=0.1, max_value=200.0,
                                         allow_nan=False, allow_infinity=False))
        rewards_np = np.random.uniform(low, high, size=n).astype(np.float32)
    elif dist_type == "skewed":
        # Exponential-like skewed rewards
        scale = data.draw(st.floats(min_value=0.1, max_value=50.0,
                                    allow_nan=False, allow_infinity=False))
        rewards_np = (np.random.exponential(scale, size=n)).astype(np.float32)
    else:
        # Spread: moderate base + perturbation (tests near-constant rewards)
        # Keep base small enough to avoid float32 catastrophic cancellation
        base = data.draw(st.floats(min_value=-10.0, max_value=10.0,
                                   allow_nan=False, allow_infinity=False))
        perturbation = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
        rewards_np = (base + perturbation).astype(np.float32)

    rewards = mx.array(rewards_np)

    # Skip if variance is essentially zero (normalization undefined)
    std_raw = float(mx.sqrt(mx.mean((rewards - mx.mean(rewards)) ** 2)))
    if std_raw < 1e-6:
        return  # degenerate case, skip

    # Normalize
    normalized = GRPOTrainer.normalize_rewards(rewards)
    mx.eval(normalized)

    norm_np = np.array(normalized)

    # Mean should be ≈ 0 (float32 tolerance)
    assert abs(np.mean(norm_np)) < 1e-3, (
        f"Normalized mean = {np.mean(norm_np)}, expected ≈ 0"
    )

    # Std should be ≈ 1 (population std, matching the implementation; float32 tolerance)
    pop_std = float(np.sqrt(np.mean((norm_np - np.mean(norm_np)) ** 2)))
    assert abs(pop_std - 1.0) < 1e-3, (
        f"Normalized std = {pop_std}, expected ≈ 1"
    )
