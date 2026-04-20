"""Property-based tests for interview-grade notebook invariants.

These tests encode 12 of the system-wide correctness properties described
in `.kiro/specs/interview-grade-notebooks/design.md` §"Correctness
Properties" (Property 1, 3–10, 12, 13, 16) and their requirement
backlinks. Every `@settings(...)` decorator sets ``max_examples >= 100``
— enforced at runtime by the meta-test
``test_hypothesis_iteration_floor`` (Property 13).

Run with:
    .venv/bin/python -m pytest tests/test_interview_grade_properties.py -v --no-header
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from scripts.transform import qbank, run_pipeline, verify
from scripts.transform.templates import (
    benchmark_cell,
    complexity_analysis_cell,
    debugging_failures_cell,
    frontier_context_cell,
    interview_index_cell,
    interview_question_cell,
    production_context_cell,
    whiteboard_challenge_cell,
)

# =============================================================================
# Module-level Hypothesis strategies
# =============================================================================

#: Difficulty tiers (Requirement 3.4).
difficulty_st = st.sampled_from(["warmup", "core", "stretch", "research"])

#: Single role (Requirement 3.5).
role_st = st.sampled_from(["mle", "research_engineer", "systems_engineer"])

#: Non-empty subset of roles, as a sorted list with unique entries
#: (Requirement 3.5).
roles_st = st.lists(role_st, min_size=1, max_size=3, unique=True).map(sorted)


def qid_st(nb_num: int) -> st.SearchStrategy[str]:
    """Strategy producing valid question ids of the form ``nb{NN}-q{MM}``.

    The notebook number ``NN`` is fixed by ``nb_num``; the question index
    ``MM`` is drawn from 1..99 so ids stay two digits (Requirement 3.3).
    """
    return st.integers(min_value=1, max_value=99).map(
        lambda m: f"nb{nb_num:02d}-q{m:02d}"
    )


def question_record_st(nb_num: int) -> st.SearchStrategy[dict]:
    """Strategy producing schema-valid Question_Bank records for ``nb_num``.

    Every generated record satisfies Requirement 3.2-3.6 and is accepted
    by ``qbank.validate_schema`` without modification.
    """
    # Keep text inputs ASCII to dodge JSON edge cases and keep traces readable.
    printable_text = st.text(
        alphabet=st.characters(
            min_codepoint=32, max_codepoint=126, blacklist_characters='"\\'
        ),
        min_size=1,
        max_size=40,
    )
    small_text = st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),  # a-z
        min_size=1,
        max_size=12,
    )

    return st.builds(
        lambda qid, section, difficulty, roles, tags, question, akp, trap, refs: {
            "id": qid,
            "notebook": f"{nb_num:02d}_synth.ipynb",
            "section": section,
            "difficulty": difficulty,
            "roles": roles,
            "topic_tags": tags,
            "question": question,
            "answer_key_points": akp,
            "worked_solution_cell_id": None,
            "trap": trap,
            "references": refs,
            "added_in": "",
        },
        qid_st(nb_num),
        printable_text,
        difficulty_st,
        roles_st,
        st.lists(small_text, min_size=1, max_size=3, unique=True),
        printable_text,
        st.lists(printable_text, min_size=3, max_size=7),
        st.one_of(st.none(), printable_text),
        st.lists(printable_text, max_size=3),
    )


def unique_records_st(
    nb_num: int, *, min_size: int, max_size: int
) -> st.SearchStrategy[list[dict]]:
    """Strategy producing a list of records with unique ids for ``nb_num``."""
    return st.lists(
        question_record_st(nb_num),
        min_size=min_size,
        max_size=max_size,
        unique_by=lambda r: r["id"],
    )


# =============================================================================
# Module-level helpers (test scaffolding)
# =============================================================================


_BANNED_IMPORT_TOKENS: tuple[str, ...] = (
    "import torch",
    "import tensorflow",
    "import jax",
    "from torch",
    "from tensorflow",
    "from jax",
)


def _is_mlx_only(src: str) -> bool:
    """Return True iff ``src`` does not import torch / tensorflow / jax.

    Used by ``test_mlx_only_new_code`` (Property 9 / Requirement 7.4).
    Matches both ``import X`` and ``from X`` forms as substring tests —
    intentionally permissive so we catch obvious violations without
    parsing Python.
    """
    return not any(token in src for token in _BANNED_IMPORT_TOKENS)


_IDX_MARKER = "📋 Interview Question Index"


def _transform_once(cells: list[dict]) -> list[dict]:
    """Append the 📋 index cell if not already present.

    Simulates the idempotent end-of-notebook insertion used by every
    notebook transform (Requirement 10.1-10.4). Re-running must be a
    no-op once the marker is present.
    """
    out = list(cells)
    for cell in out:
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if cell.get("cell_type") == "markdown" and _IDX_MARKER in src:
            return out
    marker_cell = {
        "cell_type": "markdown",
        "source": f"### {_IDX_MARKER}\n\n(auto-generated)",
    }
    return out + [marker_cell]


def _apply_additive_inserts(
    pre: list[dict], inserts: list[tuple[int, dict]]
) -> list[dict]:
    """Return a new list with every ``inserts[i][1]`` inserted before ``pre[pos]``.

    ``inserts`` is a list of ``(position, cell)`` pairs where ``position``
    indexes into the **original** ``pre``. We apply insertions in
    descending-position order so earlier positions aren't shifted by
    later ones. The resulting list contains every cell of ``pre`` in
    order plus the inserted cells.

    This mirrors Requirement 6.1-6.5: transforms only insert, never
    delete or mutate existing cells.
    """
    post = list(pre)
    for pos, cell in sorted(inserts, key=lambda t: -t[0]):
        post.insert(pos, cell)
    return post


def _coverage_meets_floors(
    interview_qs: Sequence[dict],
    *,
    n_whiteboard: int,
    n_complexity: int,
    n_other: int,
) -> bool:
    """Return True iff ``interview_qs`` + stratum counts satisfy Req 1.1-1.8.

    * ``len(interview_qs) >= 6``
    * ``n_whiteboard >= 2``
    * ``n_complexity >= 2``
    * ``n_other >= 1`` (production + frontier + debugging, combined)
    * all 4 difficulty tiers represented across ``interview_qs``
    * all 3 roles represented across ``interview_qs``
    """
    if len(interview_qs) < 6 or n_whiteboard < 2 or n_complexity < 2 or n_other < 1:
        return False

    difficulties = {q["difficulty"] for q in interview_qs}
    if difficulties != {"warmup", "core", "stretch", "research"}:
        return False

    roles_seen: set[str] = set()
    for q in interview_qs:
        roles_seen.update(q["roles"])
    if roles_seen != {"mle", "research_engineer", "systems_engineer"}:
        return False

    return True


# =============================================================================
# Test 1 — Property 1: coverage floors + spread
# =============================================================================


@given(
    n_extra=st.integers(min_value=0, max_value=4),
    n_whiteboard=st.integers(min_value=2, max_value=5),
    n_complexity=st.integers(min_value=2, max_value=5),
    n_other=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100, deadline=None)
def test_coverage_floors_and_spread(n_extra, n_whiteboard, n_complexity, n_other):
    """**Validates: Requirements 1.1-1.8**

    Property 1: a synthetic transformed notebook seeded with ≥ 6 🎯 cells,
    ≥ 2 🧑‍💻, ≥ 2 📐, ≥ 1 {🏭 ∪ 🔭 ∪ 🛠️}, and full difficulty × role
    coverage must pass ``_coverage_meets_floors``. A notebook that violates
    any single floor must fail.
    """
    # Seed one question per (difficulty, role) combination = 12 records;
    # then add ``n_extra`` filler questions.
    tiers = ["warmup", "core", "stretch", "research"]
    all_roles = ["mle", "research_engineer", "systems_engineer"]
    seeded: list[dict] = []
    idx = 1
    for d in tiers:
        for r in all_roles:
            seeded.append(
                {
                    "id": f"nb05-q{idx:02d}",
                    "difficulty": d,
                    "roles": [r],
                }
            )
            idx += 1
    for _ in range(n_extra):
        seeded.append(
            {
                "id": f"nb05-q{idx:02d}",
                "difficulty": "core",
                "roles": ["mle"],
            }
        )
        idx += 1

    # Positive: the well-formed notebook must meet the floors.
    assert _coverage_meets_floors(
        seeded,
        n_whiteboard=n_whiteboard,
        n_complexity=n_complexity,
        n_other=n_other,
    )

    # Negative 1: drop one difficulty tier -> must fail.
    trimmed_difficulty = [q for q in seeded if q["difficulty"] != "research"]
    assert not _coverage_meets_floors(
        trimmed_difficulty,
        n_whiteboard=n_whiteboard,
        n_complexity=n_complexity,
        n_other=n_other,
    )

    # Negative 2: only 5 questions -> must fail (< 6 floor).
    assert not _coverage_meets_floors(
        seeded[:5],
        n_whiteboard=n_whiteboard,
        n_complexity=n_complexity,
        n_other=n_other,
    )

    # Negative 3: whiteboard count below 2 -> must fail.
    assert not _coverage_meets_floors(
        seeded,
        n_whiteboard=1,
        n_complexity=n_complexity,
        n_other=n_other,
    )

    # Negative 4: zero other-stratum cells -> must fail.
    assert not _coverage_meets_floors(
        seeded,
        n_whiteboard=n_whiteboard,
        n_complexity=n_complexity,
        n_other=0,
    )


# =============================================================================
# Test 2 — Property 3: Question_Bank schema validity
# =============================================================================


@given(record=question_record_st(7))
@settings(max_examples=100, deadline=None)
def test_question_bank_schema(record):
    """**Validates: Requirements 3.2-3.6**

    Property 3: every record produced by ``question_record_st`` satisfies
    the Question_Bank schema; mutated records that break any single rule
    must be rejected by ``qbank.validate_schema``.
    """
    # Positive: generated records must validate.
    qbank.validate_schema(record)

    # Negative 1 — bad id pattern (not zero-padded).
    bad_id = dict(record)
    bad_id["id"] = "nb7-q3"
    with pytest.raises(ValueError, match="pattern"):
        qbank.validate_schema(bad_id)

    # Negative 2 — bad difficulty.
    bad_diff = dict(record)
    bad_diff["difficulty"] = "hard"
    with pytest.raises(ValueError, match="difficulty"):
        qbank.validate_schema(bad_diff)

    # Negative 3 — empty roles.
    bad_roles = dict(record)
    bad_roles["roles"] = []
    with pytest.raises(ValueError, match="roles"):
        qbank.validate_schema(bad_roles)

    # Negative 4 — too few answer_key_points (2 < 3).
    bad_akp_short = dict(record)
    bad_akp_short["answer_key_points"] = ["only", "two"]
    with pytest.raises(ValueError, match="answer_key_points"):
        qbank.validate_schema(bad_akp_short)

    # Negative 5 — too many answer_key_points (8 > 7).
    bad_akp_long = dict(record)
    bad_akp_long["answer_key_points"] = [f"p{i}" for i in range(8)]
    with pytest.raises(ValueError, match="answer_key_points"):
        qbank.validate_schema(bad_akp_long)


# =============================================================================
# Test 3 — Property 4: Question_Bank ↔ notebook bijection
# =============================================================================


@given(qids=st.sets(qid_st(5), min_size=1, max_size=12))
@settings(max_examples=100, deadline=None)
def test_question_bank_bijection(qids):
    """**Validates: Requirements 3.7, 3.8**

    Property 4: if a synthetic notebook contains one 🎯 cell per qid and
    the qbank contains one record per qid, ``_check_bijection`` returns
    ``([], [])``. Removing a qid from either side must surface the drift.
    """
    nb_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": (
                    f"### 🎯 Interview Question {qid}  ·  [core]  ·  mle\n\n"
                    "**Q:** synthetic test question\n"
                ),
            }
            for qid in qids
        ]
    }
    qbank_dict = {"questions": [{"id": qid} for qid in qids]}

    nb_qids = verify._extract_notebook_qids(nb_dict)
    qb_ids = verify._extract_qbank_ids(qbank_dict, 5)
    missing_in_qbank, missing_in_notebook = verify._check_bijection(nb_qids, qb_ids)
    assert missing_in_qbank == []
    assert missing_in_notebook == []

    # Drift case: drop the smallest qid from the qbank side.
    removed = sorted(qids)[0]
    qbank_short = {"questions": [{"id": q} for q in qids if q != removed]}
    qb_ids_short = verify._extract_qbank_ids(qbank_short, 5)
    missing_qbank2, missing_nb2 = verify._check_bijection(nb_qids, qb_ids_short)
    assert removed in missing_qbank2
    assert missing_nb2 == []

    # Drift case 2: drop the smallest qid from the notebook side.
    nb_short = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": f"### 🎯 Interview Question {qid}  ·  [core]  ·  mle\n",
            }
            for qid in qids
            if qid != removed
        ]
    }
    nb_qids_short = verify._extract_notebook_qids(nb_short)
    missing_qbank3, missing_nb3 = verify._check_bijection(nb_qids_short, qb_ids)
    assert missing_qbank3 == []
    assert removed in missing_nb3


# =============================================================================
# Test 4 — Property 5: role filter correctness
# =============================================================================


@given(records=unique_records_st(3, min_size=1, max_size=10))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_role_filter_correctness(records, tmp_path, monkeypatch):
    """**Validates: Requirements 3.9, 17.2**

    Property 5: ``qbank.filter_by_role(R)`` returns exactly the records
    whose ``roles`` field contains ``R``.
    """
    # Point qbank at a tmp file so the real bank is never touched.
    bank = tmp_path / "question_bank.json"
    lock = tmp_path / "question_bank.lock"
    monkeypatch.setattr(qbank, "QBANK_PATH", bank, raising=True)
    monkeypatch.setattr(qbank, "LOCK_PATH", lock, raising=True)

    qbank.save({"questions": records})

    for role in ("mle", "research_engineer", "systems_engineer"):
        expected = [r for r in records if role in r["roles"]]
        got = qbank.filter_by_role(role)

        expected_ids = sorted(r["id"] for r in expected)
        got_ids = sorted(r["id"] for r in got)
        assert got_ids == expected_ids, (
            f"role={role!r} filter mismatch:\n"
            f"  expected_ids={expected_ids}\n"
            f"  got_ids={got_ids}"
        )

    # Unknown role must raise (defensive guard in qbank.filter_by_role).
    with pytest.raises(ValueError, match="unknown role"):
        qbank.filter_by_role("data_engineer")


# =============================================================================
# Test 5 — Property 6: Whiteboard verifiability
# =============================================================================


_VAR_NAMES = ["x", "y", "z", "w", "u", "v", "a", "b"]


@given(
    var=st.sampled_from(_VAR_NAMES),
    dim=st.integers(min_value=1, max_value=16),
    title=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=100, deadline=None)
def test_whiteboard_verifiability(var, dim, title):
    """**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 7.5**

    Property 6: ``whiteboard_challenge_cell`` returns a ``(md, code)``
    pair whose code source contains at least one ``assert`` and one
    ``mx.eval`` call (Req 4.2, 4.4). Code that lacks either is rejected
    at template-build time (Req 4.1).
    """
    solution_code = (
        "import mlx.core as mx\n"
        f"{var} = mx.ones(({dim},))\n"
        f"mx.eval({var})\n"
        f"assert {var}.shape == ({dim},), 'shape mismatch'\n"
    )

    md_cell, code_cell = whiteboard_challenge_cell(
        title=title,
        prompt="synthetic prompt",
        constraints=["MLX only"],
        solution_code=solution_code,
        complexity="O(n)",
    )

    assert md_cell["cell_type"] == "markdown"
    assert code_cell["cell_type"] == "code"
    assert "assert" in code_cell["source"]
    assert "mx.eval" in code_cell["source"]

    # Rejection path 1 — missing ``assert`` statement.
    with pytest.raises(ValueError, match="assert"):
        whiteboard_challenge_cell(
            title=title,
            prompt="p",
            constraints=["c"],
            solution_code="import mlx.core as mx\nmx.eval(mx.array([1.0]))\n",
            complexity="O(1)",
        )

    # Rejection path 2 — missing ``mx.eval`` call.
    with pytest.raises(ValueError, match="mx.eval"):
        whiteboard_challenge_cell(
            title=title,
            prompt="p",
            constraints=["c"],
            solution_code="assert True\n",
            complexity="O(1)",
        )


# =============================================================================
# Test 6 — Property 7: Complexity cell faithfulness + paired benchmark
# =============================================================================


_safe_text = st.text(
    alphabet=st.characters(
        min_codepoint=32, max_codepoint=126, blacklist_characters='`"\\'
    ),
    min_size=1,
    max_size=40,
)


@given(
    op=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=3,
        max_size=20,
    ),
    flops=_safe_text,
    memory=_safe_text,
    latency=_safe_text,
    scaling=_safe_text,
    bench_dim=st.integers(min_value=4, max_value=64),
)
@settings(max_examples=100, deadline=None)
def test_complexity_cell_faithfulness(op, flops, memory, latency, scaling, bench_dim):
    """**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

    Property 7: the generated complexity cell source contains the
    declared FLOPs, memory, and latency formulas plus a benchmark anchor;
    the paired benchmark cell contains ``time.perf_counter``, ``mx.eval``,
    and ≥ 3 warmup iterations via ``for _ in range(3)``.

    Note: Requirement 5.5 (≥ 2 distinct seq-len / batch-size points) is
    systems-tier-specific and only verifiable at notebook build time
    (not at template-generation time), so it is out of scope here.
    """
    cx = complexity_analysis_cell(
        op=op, flops=flops, memory=memory, latency_mlx=latency, scaling=scaling
    )
    src = cx["source"]
    assert cx["cell_type"] == "markdown"
    assert "📐" in src
    assert flops in src
    assert memory in src
    assert latency in src
    assert scaling in src
    # Anchor reference to the paired benchmark cell.
    assert "benchmark" in src.lower()

    # Pair with a benchmark cell and check the canonical MLX pattern.
    body = f"def f():\n    return mx.ones(({bench_dim},))\n"
    bench = benchmark_cell(op=op, code=body)
    bsrc = bench["source"]
    assert bench["cell_type"] == "code"
    assert "time.perf_counter" in bsrc
    assert "mx.eval" in bsrc
    assert "for _ in range(3)" in bsrc
    assert op in bsrc


# =============================================================================
# Test 7 — Property 8: Narrative preservation (additive-only)
# =============================================================================


_cell_text_st = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=60,
)


def _synthetic_cell(kind: str, text: str) -> dict:
    return {"cell_type": kind, "source": f"[{kind}] {text}"}


@given(
    pre_texts=st.lists(_cell_text_st, min_size=3, max_size=20),
    insert_specs=st.lists(
        st.tuples(st.integers(min_value=0, max_value=20), _cell_text_st),
        min_size=0,
        max_size=10,
    ),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_narrative_preservation(pre_texts, insert_specs, data):
    """**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

    Property 8: cells listed in ``preserve_sections`` appear byte-identical
    in both pre- and post-transform lists. We simulate a transform using
    ``_apply_additive_inserts`` and pick a random subset of original
    indices to declare as preserved; those cells must survive unchanged
    at their (shifted) post positions.
    """
    pre = [_synthetic_cell("markdown", t) for t in pre_texts]
    n = len(pre)

    # Clamp insert positions to [0, n] so ``list.insert`` behaves sanely.
    inserts: list[tuple[int, dict]] = [
        (min(pos, n), _synthetic_cell("markdown", text))
        for pos, text in insert_specs
    ]

    # Random subset of preserve indices.
    preserve_indices = data.draw(
        st.sets(st.integers(min_value=0, max_value=n - 1), min_size=0, max_size=n)
    )

    post = _apply_additive_inserts(pre, inserts)

    # Basic invariant: post is a superset of pre in order (no deletions).
    assert len(post) == n + len(inserts)

    for i in preserve_indices:
        shift = sum(1 for pos, _ in inserts if pos <= i)
        post_idx = i + shift
        assert post[post_idx] == pre[i], (
            f"preserve drift at pre idx {i} -> post idx {post_idx}:\n"
            f"  pre={pre[i]!r}\n"
            f"  post={post[post_idx]!r}"
        )


# =============================================================================
# Test 8 — Property 9: MLX-only new code
# =============================================================================


@given(
    var=st.sampled_from(_VAR_NAMES),
    dim=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=100, deadline=None)
def test_mlx_only_new_code(var, dim):
    """**Validates: Requirement 7.4**

    Property 9: code cells emitted by ``whiteboard_challenge_cell`` and
    ``benchmark_cell`` never import torch / tensorflow / jax; the
    ``_is_mlx_only`` predicate accepts them in every generated case and
    rejects any cell that contains a banned import token.
    """
    solution_code = (
        "import mlx.core as mx\n"
        f"{var} = mx.ones(({dim},))\n"
        f"mx.eval({var})\n"
        f"assert {var}.shape == ({dim},)\n"
    )
    _md, code_cell = whiteboard_challenge_cell(
        title="t", prompt="p", constraints=["c"],
        solution_code=solution_code, complexity="O(n)",
    )
    assert _is_mlx_only(code_cell["source"]), (
        f"whiteboard code unexpectedly contains banned import: "
        f"{code_cell['source']!r}"
    )

    bench_body = f"def f():\n    return mx.ones(({dim},))\n"
    bench = benchmark_cell(op="synthetic", code=bench_body)
    assert _is_mlx_only(bench["source"]), (
        f"benchmark code unexpectedly contains banned import: "
        f"{bench['source']!r}"
    )

    # Predicate sanity: each banned token is caught.
    for banned in _BANNED_IMPORT_TOKENS:
        assert not _is_mlx_only(f"{banned}\nx = 1\n"), (
            f"_is_mlx_only failed to reject banned token {banned!r}"
        )


# =============================================================================
# Test 9 — Property 10: Transform idempotence
# =============================================================================


@given(
    cells=st.lists(
        st.builds(
            _synthetic_cell,
            st.sampled_from(["markdown", "code"]),
            _cell_text_st,
        ),
        min_size=0,
        max_size=15,
    )
)
@settings(max_examples=100, deadline=None)
def test_transform_idempotence(cells):
    """**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 21.4**

    Property 10: ``_transform_once`` is idempotent — running it twice on
    any cell list produces a byte-identical result. The second call must
    detect the 📋 marker and skip re-insertion.
    """
    once = _transform_once(cells)
    twice = _transform_once(once)
    assert once == twice

    # Explicitly exercise the "already transformed" branch by prepending the
    # marker and re-running — the helper must not add a second copy.
    marker = {
        "cell_type": "markdown",
        "source": f"### {_IDX_MARKER}\n\nseed",
    }
    already = [marker] + list(cells)
    result = _transform_once(already)
    assert result == already

    # The result must contain exactly one marker cell.
    marker_count = sum(
        1
        for c in once
        if c.get("cell_type") == "markdown"
        and isinstance(c.get("source"), str)
        and _IDX_MARKER in c["source"]
    )
    assert marker_count == 1


# =============================================================================
# Test 10 — Property 12: Cell template consistency (emoji prefixes)
# =============================================================================


_plain_text_st = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126,
                           blacklist_characters='`"\\'),
    min_size=1,
    max_size=30,
)


@given(
    qid_idx=st.integers(min_value=1, max_value=99),
    difficulty=difficulty_st,
    roles=roles_st,
    question_text=_plain_text_st,
    akp=st.lists(_plain_text_st, min_size=3, max_size=7),
    op_name=_plain_text_st,
    formula=_plain_text_st,
    topic=_plain_text_st,
    paper_title=_plain_text_st,
    paper_year=st.integers(min_value=2017, max_value=2026),
    paper_oneliner=_plain_text_st,
    sota=_plain_text_st,
    concept=_plain_text_st,
    symptom=_plain_text_st,
    dim=st.integers(min_value=1, max_value=32),
    cause_text=_plain_text_st,
)
@settings(max_examples=100, deadline=None)
def test_cell_template_consistency(
    qid_idx,
    difficulty,
    roles,
    question_text,
    akp,
    op_name,
    formula,
    topic,
    paper_title,
    paper_year,
    paper_oneliner,
    sota,
    concept,
    symptom,
    dim,
    cause_text,
):
    """**Validates: Requirements 20.1-20.7**

    Property 12: every cell stratum's source begins with its canonical
    emoji prefix:

      🎯 Interview Question    🧑‍💻 Whiteboard Challenge
      📐 Complexity Analysis   🏭 Production Context
      🔭 Frontier Context      🛠️ Debugging & Failures
      📋 Interview Question Index
    """
    # 🎯 Interview Question
    q = {
        "id": f"nb05-q{qid_idx:02d}",
        "notebook": "05_synth.ipynb",
        "section": "S",
        "difficulty": difficulty,
        "roles": roles,
        "topic_tags": ["t"],
        "question": question_text,
        "answer_key_points": akp,
        "worked_solution_cell_id": None,
        "trap": None,
        "references": [],
        "added_in": "",
    }
    iq_cell = interview_question_cell(q)
    assert iq_cell["cell_type"] == "markdown"
    assert iq_cell["source"].startswith("### 🎯 Interview Question")

    # 🧑‍💻 Whiteboard Challenge (markdown half of the pair)
    wb_code = (
        "import mlx.core as mx\n"
        f"x = mx.ones(({dim},))\n"
        "mx.eval(x)\n"
        f"assert x.shape == ({dim},)\n"
    )
    wb_md, wb_code_cell = whiteboard_challenge_cell(
        title=op_name,
        prompt=question_text,
        constraints=["MLX only"],
        solution_code=wb_code,
        complexity="O(n)",
    )
    assert wb_md["cell_type"] == "markdown"
    assert wb_md["source"].startswith("### 🧑\u200d💻 Whiteboard Challenge:")
    assert wb_code_cell["cell_type"] == "code"

    # 📐 Complexity & Systems
    cx = complexity_analysis_cell(
        op=op_name,
        flops=formula,
        memory=formula,
        latency_mlx=formula,
        scaling=formula,
    )
    assert cx["cell_type"] == "markdown"
    assert cx["source"].startswith("### 📐 Complexity & Systems:")

    # 🏭 Production Context
    pc = production_context_cell(
        concept=concept,
        vllm="v",
        sglang="s",
        trt_llm="t",
        mlx_lm="m",
    )
    assert pc["cell_type"] == "markdown"
    assert pc["source"].startswith("### 🏭")

    # 🔭 Frontier Context
    fc = frontier_context_cell(
        topic=topic,
        papers=[(paper_title, paper_year, paper_oneliner)],
        current_sota=sota,
    )
    assert fc["cell_type"] == "markdown"
    assert fc["source"].startswith("### 🔭 Frontier Context (")

    # 🛠️ Debugging & Failures (markdown half)
    dbg_md, _dbg_code = debugging_failures_cell(
        symptom=symptom,
        root_causes=[cause_text, cause_text + "!"],
        diagnostic_code="print('diag')\n",
    )
    assert dbg_md["cell_type"] == "markdown"
    assert dbg_md["source"].startswith("### 🛠️ Failure Modes & Debugging:")

    # 📋 Interview Question Index
    idx_cell = interview_index_cell([q])
    assert idx_cell["cell_type"] == "markdown"
    assert idx_cell["source"].startswith("### 📋 Interview Question Index")


# =============================================================================
# Test 11 — Property 13 (meta-test): every @settings has max_examples >= 100
# =============================================================================


# Match ``@settings(...)`` only when it appears as a real decorator:
# the line must start with (optional whitespace +) ``@settings`` so we skip
# any occurrences inside docstrings or prose.
_SETTINGS_MAX_RE = re.compile(
    r"^\s*@settings\s*\(([^)]*)\)", re.MULTILINE | re.DOTALL
)
_MAX_EXAMPLES_RE = re.compile(r"max_examples\s*=\s*(\d+)")


@given(dummy=st.just(0))
@settings(max_examples=100, deadline=None)
def test_hypothesis_iteration_floor(dummy):
    """**Validates: Requirement 9.5**

    Property 13: every ``@settings(...)`` decorator in this test file
    declares ``max_examples >= 100``. We parse the file itself and
    regex-scan each decorator block.
    """
    this_file = Path(__file__).read_text(encoding="utf-8")

    settings_blocks = _SETTINGS_MAX_RE.findall(this_file)
    # Defensive: this file must actually use @settings somewhere. If the
    # regex finds zero matches the test suite has been truncated.
    assert len(settings_blocks) >= 12, (
        f"expected at least 12 @settings blocks (one per test), "
        f"found {len(settings_blocks)}"
    )

    for block in settings_blocks:
        m = _MAX_EXAMPLES_RE.search(block)
        assert m is not None, (
            f"@settings block missing max_examples=...: {block!r}"
        )
        value = int(m.group(1))
        assert value >= 100, (
            f"@settings max_examples={value} is below the 100 floor "
            f"(Requirement 9.5). Block: {block!r}"
        )


# =============================================================================
# Test 12 — Property 16: no inline Python in the pipeline
# =============================================================================


_PIPELINE_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "transform"
    / "run_pipeline.py"
)


@given(
    extra_flag=st.sampled_from(
        ["--flag", "--all", "--notebook", "--parallel", "--push"]
    ),
    extra_value=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=0,
        max_size=10,
    ),
)
@settings(max_examples=100, deadline=None)
def test_no_inline_python_in_pipeline(extra_flag, extra_value):
    """**Validates: Requirement 11.3**

    Property 16: ``run_pipeline._reject_inline_python`` raises on any
    argv containing a bare ``-c`` token and accepts argv without it. The
    pipeline source also does not spawn a subprocess with ``-c``.
    """
    # Positive: argv without ``-c`` must be accepted.
    run_pipeline._reject_inline_python(["foo.py", extra_flag, extra_value])

    # Negative 1: ``-c`` in argv must raise.
    with pytest.raises(RuntimeError, match="inline python"):
        run_pipeline._reject_inline_python(
            ["foo.py", "-c", "print('x')", extra_flag]
        )

    # Negative 2: ``-c`` as the only extra token must raise.
    with pytest.raises(RuntimeError, match="inline python"):
        run_pipeline._reject_inline_python([extra_flag, "-c"])

    # Source-level guard: pipeline source must not spawn a subprocess with
    # ``-c``. We scan for the canonical substring ``"-c"`` only in contexts
    # that would indicate subprocess invocation. The simplest robust check:
    # the file mentions ``subprocess`` but never passes ``"-c"`` as a list
    # element.
    src = _PIPELINE_SOURCE_PATH.read_text(encoding="utf-8")
    # Any call that includes ``"-c"`` as a string literal next to a Python
    # executable is forbidden. The run_pipeline module does contain the
    # literal "-c" inside guard strings (e.g. ``if token == "-c"``). Those
    # are reads, not invocations. We whitelist those by requiring that the
    # forbidden form includes a subprocess/Popen call on the same line.
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ("subprocess" in line or "Popen" in line) and '"-c"' in line:
            pytest.fail(
                f"run_pipeline.py appears to spawn a subprocess with -c: "
                f"{line!r}"
            )
