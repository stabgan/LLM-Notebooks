"""Unit tests for the pipeline runner and commit helpers.

These tests cover the small, pure helpers that carry the load-bearing
logic of the pipeline entry point:

* ``run_pipeline._reject_inline_python`` — enforces Requirement 11.3.
* ``run_pipeline._resolve_notebooks``   — argparse → nb list resolution.
* ``commit._build_message``             — conventional-commit message format
  (Requirement 16.3).

Run with:
    .venv/bin/python -m pytest tests/test_pipeline.py -v --no-header
"""

from __future__ import annotations

import pytest

from scripts.transform import commit, run_pipeline


# ---------------------------------------------------------------------------
# run_pipeline._reject_inline_python (Requirement 11.3)
# ---------------------------------------------------------------------------


def test_reject_inline_python_allows_normal_argv():
    # Must not raise for a clean argv.
    run_pipeline._reject_inline_python(["foo.py", "--flag"])


def test_reject_inline_python_allows_substring_not_exact_match():
    # Flags that merely contain "c" should not trip the guard.
    run_pipeline._reject_inline_python(
        ["foo.py", "--check", "-csv", "--config", "cat"]
    )


def test_reject_inline_python_raises_on_dash_c():
    with pytest.raises(RuntimeError, match="inline python"):
        run_pipeline._reject_inline_python(["foo.py", "-c", "print(1)"])


def test_reject_inline_python_raises_when_dash_c_is_anywhere():
    with pytest.raises(RuntimeError, match="inline python"):
        run_pipeline._reject_inline_python(["foo.py", "--flag", "-c"])


# ---------------------------------------------------------------------------
# run_pipeline._resolve_notebooks
# ---------------------------------------------------------------------------


def test_resolve_notebooks_all_returns_zero_through_nineteen():
    assert run_pipeline._resolve_notebooks(all=True, notebook=None) == list(
        range(0, 20)
    )


def test_resolve_notebooks_single_returns_singleton_list():
    assert run_pipeline._resolve_notebooks(all=False, notebook=5) == [5]


def test_resolve_notebooks_requires_notebook_when_not_all():
    with pytest.raises(ValueError, match="notebook"):
        run_pipeline._resolve_notebooks(all=False, notebook=None)


def test_resolve_notebooks_rejects_out_of_range():
    with pytest.raises(ValueError, match="notebook"):
        run_pipeline._resolve_notebooks(all=False, notebook=-1)
    with pytest.raises(ValueError, match="notebook"):
        run_pipeline._resolve_notebooks(all=False, notebook=100)


# ---------------------------------------------------------------------------
# commit._build_message (Requirement 16.3)
# ---------------------------------------------------------------------------


def test_build_message_formats_two_digit_nb_and_stage():
    assert commit._build_message(7, "test stage") == "feat(nb07): test stage"


def test_build_message_zero_pads_single_digit():
    assert commit._build_message(0, "foo") == "feat(nb00): foo"
    assert commit._build_message(9, "bar") == "feat(nb09): bar"


def test_build_message_preserves_two_digit_nb():
    assert commit._build_message(19, "final") == "feat(nb19): final"


def test_build_message_strips_surrounding_whitespace_in_stage():
    assert (
        commit._build_message(3, "  padded stage  ")
        == "feat(nb03): padded stage"
    )


def test_build_message_rejects_empty_stage():
    with pytest.raises(ValueError, match="stage"):
        commit._build_message(3, "")
    with pytest.raises(ValueError, match="stage"):
        commit._build_message(3, "   ")


def test_build_message_rejects_out_of_range_nb():
    with pytest.raises(ValueError, match="nb_num"):
        commit._build_message(-1, "foo")
    with pytest.raises(ValueError, match="nb_num"):
        commit._build_message(100, "foo")
