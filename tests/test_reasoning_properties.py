"""Property-based tests for Reasoning & Test-Time Compute (Notebook 19).

**Validates: Requirement 5.5**

Property 15: Process Reward Model Bounds — scores in [0, 1] for any input.
"""

import mlx.core as mx
from hypothesis import given, settings, strategies as st

from utils.reasoning import ProcessRewardModel


# ---------------------------------------------------------------------------
# Strategies — generate arbitrary text strings for context and step
# ---------------------------------------------------------------------------

# Arbitrary text strings (printable ASCII, varying lengths)
text_st = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=200,
)


# ---------------------------------------------------------------------------
# Property 15: Process Reward Model Bounds
# ---------------------------------------------------------------------------

@given(
    context=text_st,
    step=text_st,
)
@settings(deadline=None, max_examples=50)
def test_process_reward_model_bounds(context, step):
    """**Validates: Requirement 5.5**

    Property 15: For any reasoning step and context, the Process_Reward_Model
    SHALL return a score in the range [0, 1].

    The sigmoid output layer guarantees this mathematically, but we verify
    it holds across arbitrary inputs including empty strings, long strings,
    and strings with special characters.
    """
    prm = ProcessRewardModel(vocab_size=256, d_model=32, d_hidden=64, max_len=64)

    score = prm.score_step(context, step)
    mx.eval(score)
    score_val = score.item()

    assert 0.0 <= score_val <= 1.0, (
        f"PRM score out of bounds: {score_val:.6f} for "
        f"context={context!r:.50}, step={step!r:.50}"
    )


@given(
    context=text_st,
    step=text_st,
    d_model=st.sampled_from([16, 32, 64]),
    d_hidden=st.sampled_from([32, 64, 128]),
)
@settings(deadline=None, max_examples=30)
def test_process_reward_model_bounds_various_sizes(context, step, d_model, d_hidden):
    """**Validates: Requirement 5.5**

    Property 15 (extended): PRM scores remain in [0, 1] across different
    model configurations (varying d_model and d_hidden).
    """
    prm = ProcessRewardModel(
        vocab_size=256, d_model=d_model, d_hidden=d_hidden, max_len=64
    )

    score = prm.score_step(context, step)
    mx.eval(score)
    score_val = score.item()

    assert 0.0 <= score_val <= 1.0, (
        f"PRM score out of bounds: {score_val:.6f} "
        f"(d_model={d_model}, d_hidden={d_hidden})"
    )
