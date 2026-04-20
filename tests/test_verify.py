"""Smoke tests for `scripts.transform.verify` helper functions.

We test the small, pure helpers in isolation and avoid running the full
subprocess-based `verify_notebook` here — that path shells out to
`nbconvert` + `pytest` and is covered by integration runs. The helpers
below carry the load-bearing logic:

  * `_extract_notebook_qids(notebook_dict)`
  * `_extract_qbank_ids(qbank_dict, nb_num)`
  * `_check_bijection(nb_qids, qbank_ids)`
  * `_parse_pytest_passing(output)`
  * `_extract_failing_cell_index(stdout, stderr)`
  * `_parse_notebooks_arg(spec)`
  * `_locate_notebook(nb_num)` (glob semantics only)

Run with:
    .venv/bin/python -m pytest tests/test_verify.py -v --no-header
"""

from __future__ import annotations

import pytest

from scripts.transform import verify


# ---------------------------------------------------------------------------
# _extract_notebook_qids
# ---------------------------------------------------------------------------


def _nb(cells: list[dict]) -> dict:
    return {"cells": cells}


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "source": source}


def _code(source: str) -> dict:
    return {"cell_type": "code", "source": source}


def test_extract_notebook_qids_returns_sorted_unique_ids():
    nb = _nb(
        [
            _md("# Intro\nsome prose"),
            _md(
                "### 🎯 Interview Question nb05-q03  ·  [core]  ·  mle\n"
                "**Q:** What..."
            ),
            _code("import mlx.core as mx\n"),
            _md("### 🎯 Interview Question nb05-q01  ·  [warmup]  ·  mle"),
            # Duplicate qid in a different cell — should dedupe.
            _md("### 🎯 Interview Question nb05-q03  ·  [core]  ·  re"),
        ]
    )
    assert verify._extract_notebook_qids(nb) == ["nb05-q01", "nb05-q03"]


def test_extract_notebook_qids_handles_list_source_format():
    # Jupyter often stores source as a list of lines.
    nb = _nb(
        [
            {
                "cell_type": "markdown",
                "source": [
                    "### 🎯 Interview Question nb00-q02  ·  [warmup]  ·  mle\n",
                    "\n",
                    "**Q:** What is lazy evaluation?\n",
                ],
            }
        ]
    )
    assert verify._extract_notebook_qids(nb) == ["nb00-q02"]


def test_extract_notebook_qids_ignores_code_cells_even_if_matching():
    nb = _nb(
        [
            _code("# 🎯 Interview Question nb05-q03  this is a comment"),
            _md("### 🎯 Interview Question nb05-q01"),
        ]
    )
    assert verify._extract_notebook_qids(nb) == ["nb05-q01"]


def test_extract_notebook_qids_empty_notebook_returns_empty_list():
    assert verify._extract_notebook_qids({"cells": []}) == []
    assert verify._extract_notebook_qids({}) == []


def test_extract_notebook_qids_ignores_malformed_header():
    nb = _nb(
        [
            # Missing emoji
            _md("### Interview Question nb05-q03"),
            # Wrong number of digits
            _md("### 🎯 Interview Question nb5-q3"),
            # Correct header
            _md("### 🎯 Interview Question nb07-q09"),
        ]
    )
    assert verify._extract_notebook_qids(nb) == ["nb07-q09"]


# ---------------------------------------------------------------------------
# _extract_qbank_ids
# ---------------------------------------------------------------------------


def test_extract_qbank_ids_filters_by_notebook_prefix():
    qbank = {
        "questions": [
            {"id": "nb00-q01"},
            {"id": "nb05-q02"},
            {"id": "nb05-q01"},
            {"id": "nb07-q01"},
        ]
    }
    assert verify._extract_qbank_ids(qbank, 5) == ["nb05-q01", "nb05-q02"]
    assert verify._extract_qbank_ids(qbank, 0) == ["nb00-q01"]
    assert verify._extract_qbank_ids(qbank, 9) == []


def test_extract_qbank_ids_handles_empty_bank():
    assert verify._extract_qbank_ids({"questions": []}, 0) == []
    assert verify._extract_qbank_ids({}, 0) == []


# ---------------------------------------------------------------------------
# _check_bijection
# ---------------------------------------------------------------------------


def test_check_bijection_empty_both_sides_is_vacuously_true():
    missing_qbank, missing_nb = verify._check_bijection([], [])
    assert missing_qbank == []
    assert missing_nb == []


def test_check_bijection_detects_drift_on_both_sides():
    nb_qids = ["nb05-q01", "nb05-q02", "nb05-q04"]
    qbank_ids = ["nb05-q02", "nb05-q03", "nb05-q04"]
    missing_qbank, missing_nb = verify._check_bijection(nb_qids, qbank_ids)
    assert missing_qbank == ["nb05-q01"]  # in notebook, not in qbank
    assert missing_nb == ["nb05-q03"]  # in qbank, not in notebook


def test_check_bijection_perfect_match_returns_empty_lists():
    ids = ["nb05-q01", "nb05-q02", "nb05-q03"]
    missing_qbank, missing_nb = verify._check_bijection(ids, ids)
    assert missing_qbank == []
    assert missing_nb == []


def test_check_bijection_accepts_arbitrary_iterables():
    # Sets and generators should work — the helper only needs iterability.
    nb_qids = iter(["nb05-q02", "nb05-q01"])
    qbank_ids = {"nb05-q01", "nb05-q02"}
    missing_qbank, missing_nb = verify._check_bijection(nb_qids, qbank_ids)
    assert missing_qbank == []
    assert missing_nb == []


# ---------------------------------------------------------------------------
# _parse_pytest_passing
# ---------------------------------------------------------------------------


def test_parse_pytest_passing_typical_summary():
    out = "===== 34 passed, 2 skipped in 3.21s ====="
    assert verify._parse_pytest_passing(out) == 34


def test_parse_pytest_passing_failure_line_still_parses_passing_count():
    out = "===== 1 failed, 33 passed in 3.21s ====="
    assert verify._parse_pytest_passing(out) == 33


def test_parse_pytest_passing_no_passed_token_returns_none():
    assert verify._parse_pytest_passing("no tests collected") is None


# ---------------------------------------------------------------------------
# _extract_failing_cell_index
# ---------------------------------------------------------------------------


def test_extract_failing_cell_index_from_input_line_format():
    stderr = "Traceback...\nCell In[5], line 2\n  raise ValueError(...)"
    assert verify._extract_failing_cell_index("", stderr) == 5


def test_extract_failing_cell_index_returns_none_when_absent():
    assert verify._extract_failing_cell_index("", "") is None
    assert (
        verify._extract_failing_cell_index("some stdout", "some stderr")
        is None
    )


# ---------------------------------------------------------------------------
# _parse_notebooks_arg
# ---------------------------------------------------------------------------


def test_parse_notebooks_arg_range():
    assert verify._parse_notebooks_arg("00-04") == [0, 1, 2, 3, 4]


def test_parse_notebooks_arg_comma_list():
    assert verify._parse_notebooks_arg("00,05,10") == [0, 5, 10]


def test_parse_notebooks_arg_mixed_range_and_list():
    assert verify._parse_notebooks_arg("00-02,05,15-17") == [0, 1, 2, 5, 15, 16, 17]


def test_parse_notebooks_arg_deduplicates_and_sorts():
    assert verify._parse_notebooks_arg("05,00-02,01") == [0, 1, 2, 5]


def test_parse_notebooks_arg_rejects_reversed_range():
    with pytest.raises(ValueError, match="invalid range"):
        verify._parse_notebooks_arg("09-05")


# ---------------------------------------------------------------------------
# _locate_notebook — glob semantics (we inspect real files at repo root)
# ---------------------------------------------------------------------------


def test_locate_notebook_validates_nb_num_type():
    with pytest.raises(ValueError, match="nb_num"):
        verify._locate_notebook(-1)
    with pytest.raises(ValueError, match="nb_num"):
        verify._locate_notebook("05")  # type: ignore[arg-type]


def test_locate_notebook_finds_existing_notebook():
    # nb00 exists at repo root.
    p = verify._locate_notebook(0)
    assert p.name == "00_environment_apple_silicon.ipynb"


def test_locate_notebook_raises_on_missing_notebook():
    # nb99 does not exist.
    with pytest.raises(FileNotFoundError):
        verify._locate_notebook(99)
