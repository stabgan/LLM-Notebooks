"""Property-based tests for inference optimization utilities (Notebook 11).

**Validates: Requirements 9.2, 9.4, 9.7**

Property 23: KV-Cache Equivalence — cached logits identical to full recomputation.
Property 24: Quantization Error Bound — max error per group ≤ (max-min)/(2^N-1).
Property 25: Speculative Decoding Correctness — accepted tokens match target-only.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from hypothesis import given, settings, strategies as st, assume

from utils.inference import (
    SimpleLM,
    KVCacheManager,
    Quantizer,
    QuantizedTensor,
    SpeculativeDecoder,
    create_model_pair,
)


# ---------------------------------------------------------------------------
# Property 23: KV-Cache Equivalence
# ---------------------------------------------------------------------------

@given(
    vocab_size=st.sampled_from([32, 64]),
    d_model=st.sampled_from([16, 32]),
    n_layers=st.sampled_from([1, 2]),
    prompt_len=st.integers(min_value=2, max_value=8),
    gen_len=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=30)
def test_kv_cache_equivalence(vocab_size, d_model, n_layers, prompt_len, gen_len, seed):
    """**Validates: Requirement 9.2**

    Property 23: Token generation with KV-cache produces logits identical
    to full recomputation without cache within floating-point tolerance.

    We generate tokens with the KV-cache manager, then recompute logits
    for the full sequence (prompt + generated) without cache, and verify
    the logits at each generated position match.
    """
    mx.random.seed(seed)
    d_ff = d_model * 2

    model = SimpleLM(vocab_size, d_model, n_layers, d_ff)
    mx.eval(model.parameters())

    # Random prompt
    prompt_ids = mx.random.randint(0, vocab_size, shape=(1, prompt_len))
    mx.eval(prompt_ids)

    # --- Generate with cache ---
    cache_mgr = KVCacheManager(model)
    generated_ids, cached_logits_list = cache_mgr.generate(prompt_ids, gen_len)
    mx.eval(generated_ids)

    # Build full sequence: prompt + generated
    full_ids = mx.concatenate([prompt_ids, generated_ids], axis=1)
    mx.eval(full_ids)

    # --- Full recomputation without cache ---
    full_logits = KVCacheManager.generate_without_cache(model, full_ids)

    # Compare logits at each generated position
    prompt_l = prompt_ids.shape[1]
    for t in range(gen_len):
        # Cached logits for generated token t
        cached_t = np.array(cached_logits_list[t][0, 0, :])
        # Full recomputation logits at position (prompt_len - 1 + t)
        # because logits at position i predict token i+1
        full_t = np.array(full_logits[0, prompt_l - 1 + t, :])

        np.testing.assert_allclose(
            cached_t,
            full_t,
            atol=1e-4,
            rtol=1e-3,
            err_msg=(
                f"KV-cache logits mismatch at generated position {t}. "
                f"Max diff: {np.max(np.abs(cached_t - full_t)):.6f}"
            ),
        )


# ---------------------------------------------------------------------------
# Property 24: Quantization Error Bound
# ---------------------------------------------------------------------------

@given(
    rows=st.integers(min_value=1, max_value=16),
    cols=st.sampled_from([32, 64, 128]),
    bits=st.sampled_from([4, 8]),
    group_size=st.sampled_from([16, 32, 64]),
    scale_factor=st.floats(min_value=0.01, max_value=10.0),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=100)
def test_quantization_error_bound(rows, cols, bits, group_size, scale_factor, seed):
    """**Validates: Requirement 9.4**

    Property 24: For any weight tensor, bit width N, and group size G,
    the maximum absolute error per group after quantize/dequantize is
    no greater than (max_val - min_val) / (2^N - 1).

    This is the fundamental guarantee of uniform min-max quantization.
    """
    mx.random.seed(seed)

    # Generate random weights scaled by scale_factor
    weights = mx.random.normal(shape=(rows, cols)) * scale_factor
    mx.eval(weights)

    # Quantize and dequantize
    if bits == 4:
        qtensor = Quantizer.quantize_4bit(weights, group_size=group_size)
    else:
        qtensor = Quantizer.quantize_8bit(weights, group_size=group_size)

    deq = Quantizer.dequantize(qtensor)
    mx.eval(deq)

    # Compute actual error
    error = mx.abs(weights - deq)
    mx.eval(error)

    # Compute theoretical bound per group
    flat_w = weights.reshape(-1)
    n_elem = flat_w.shape[0]
    remainder = n_elem % group_size
    if remainder != 0:
        pad_size = group_size - remainder
        flat_w = mx.concatenate([flat_w, mx.zeros(pad_size)])
    grouped_w = flat_w.reshape(-1, group_size)

    flat_e = error.reshape(-1)
    if remainder != 0:
        flat_e = mx.concatenate([flat_e, mx.zeros(pad_size)])
    grouped_e = flat_e.reshape(-1, group_size)

    g_min = mx.min(grouped_w, axis=-1)
    g_max = mx.max(grouped_w, axis=-1)
    n_levels = 2 ** bits - 1
    theoretical_bound = (g_max - g_min) / n_levels
    mx.eval(theoretical_bound)

    # Max error per group
    actual_max_per_group = mx.max(grouped_e, axis=-1)
    mx.eval(actual_max_per_group)

    # Check: actual max error per group ≤ theoretical bound + small epsilon
    # (epsilon accounts for floating-point rounding in the round() operation)
    epsilon = 1e-5
    for g in range(actual_max_per_group.shape[0]):
        actual = actual_max_per_group[g].item()
        bound = theoretical_bound[g].item()
        assert actual <= bound + epsilon, (
            f"Group {g}: actual max error {actual:.8f} exceeds "
            f"theoretical bound {bound:.8f} + eps {epsilon}"
        )



# ---------------------------------------------------------------------------
# Property 25: Speculative Decoding Correctness
# ---------------------------------------------------------------------------

@given(
    vocab_size=st.sampled_from([32, 64]),
    prompt_len=st.integers(min_value=3, max_value=6),
    n_draft=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(deadline=None, max_examples=20)
def test_speculative_decoding_correctness(vocab_size, prompt_len, n_draft, seed):
    """**Validates: Requirement 9.7**

    Property 25: The tokens accepted by the speculative decoder's verify step
    are identical to the tokens the target model would generate alone.

    We run the target model greedily to get reference tokens, then run
    speculative verify and check that every accepted token matches.
    """
    mx.random.seed(seed)

    # Create draft/target pair
    draft_model = SimpleLM(vocab_size, d_model=16, n_layers=1, d_ff=32)
    target_model = SimpleLM(vocab_size, d_model=32, n_layers=2, d_ff=64)
    mx.eval(draft_model.parameters())
    mx.eval(target_model.parameters())

    # Random prompt
    prompt_ids = mx.random.randint(0, vocab_size, shape=(1, prompt_len))
    mx.eval(prompt_ids)

    # --- Reference: target-only generation ---
    ref_tokens = SpeculativeDecoder.generate_target_only(
        target_model, prompt_ids, max_new_tokens=n_draft,
    )
    mx.eval(ref_tokens)

    # --- Speculative: draft then verify ---
    draft_tokens, _ = SpeculativeDecoder.draft(draft_model, prompt_ids, n_draft)
    mx.eval(draft_tokens)

    accepted, n_accepted = SpeculativeDecoder.verify(
        target_model, prompt_ids, draft_tokens,
    )
    mx.eval(accepted)

    # Every accepted token must match the reference
    ref_np = np.array(ref_tokens[0])
    acc_np = np.array(accepted[0])

    n_to_check = min(len(acc_np), len(ref_np))
    for i in range(n_to_check):
        assert acc_np[i] == ref_np[i], (
            f"Token {i}: accepted={acc_np[i]}, reference={ref_np[i]}. "
            f"Draft tokens: {np.array(draft_tokens[0])}. "
            f"Accepted: {acc_np[:n_to_check]}. Reference: {ref_np[:n_to_check]}"
        )
