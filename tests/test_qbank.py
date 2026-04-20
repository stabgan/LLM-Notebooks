"""Smoke tests for `scripts.transform.qbank`.

Covers:
  - load/save roundtrip via a temporary bank path
  - upsert_slice + re-upsert idempotence and slice isolation
  - delete_slice
  - filter_by_role
  - validate_schema on good and bad records

The tests monkeypatch `QBANK_PATH` / `LOCK_PATH` to point at a tempdir so
the repo's real bank is never touched.

Run with:
    .venv/bin/python -m pytest tests/test_qbank.py -v --no-header
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.transform import qbank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_qbank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the qbank module to a temp file and seed it empty.

    Returns the path to the temporary question bank JSON file.
    """
    bank = tmp_path / "question_bank.json"
    lock = tmp_path / "question_bank.lock"
    monkeypatch.setattr(qbank, "QBANK_PATH", bank, raising=True)
    monkeypatch.setattr(qbank, "LOCK_PATH", lock, raising=True)

    bank.write_text('{"questions": []}\n', encoding="utf-8")
    return bank


def _sample_record(qid: str, *, roles: list[str] | None = None) -> dict:
    """Build a schema-valid record for tests."""
    return {
        "id": qid,
        "notebook": f"{qid.split('-')[0][2:]}_sample.ipynb",
        "section": "Sample section",
        "difficulty": "core",
        "roles": roles if roles is not None else ["mle", "research_engineer"],
        "topic_tags": ["sample", "unit-test"],
        "question": f"What does question {qid} ask?",
        "answer_key_points": [
            "point one",
            "point two",
            "point three",
        ],
        "worked_solution_cell_id": None,
        "trap": None,
        "references": [],
        "added_in": "",
    }


# ---------------------------------------------------------------------------
# load / save roundtrip
# ---------------------------------------------------------------------------


def test_load_returns_empty_bank_initially(tmp_qbank: Path):
    data = qbank.load()
    assert data == {"questions": []}


def test_save_and_load_roundtrip_preserves_content(tmp_qbank: Path):
    payload = {
        "questions": [_sample_record("nb00-q01"), _sample_record("nb05-q03")]
    }
    qbank.save(payload)

    # File on disk is valid JSON and matches the payload.
    raw = json.loads(tmp_qbank.read_text(encoding="utf-8"))
    assert raw == payload

    # load() reads the same structure back.
    loaded = qbank.load()
    assert loaded == payload


def test_save_is_atomic_no_tmpfile_leftover(tmp_qbank: Path):
    qbank.save({"questions": [_sample_record("nb00-q01")]})
    leftovers = list(tmp_qbank.parent.glob(".question_bank.*.tmp"))
    assert leftovers == [], f"tmp files leaked: {leftovers}"


# ---------------------------------------------------------------------------
# upsert_slice — idempotence + slice isolation
# ---------------------------------------------------------------------------


def test_upsert_slice_writes_only_target_slice(tmp_qbank: Path):
    # Seed with records belonging to nb05 and nb07.
    seed = {
        "questions": [
            _sample_record("nb05-q01"),
            _sample_record("nb05-q02"),
            _sample_record("nb07-q01"),
        ]
    }
    qbank.save(seed)

    # Replace nb05's slice; nb07 must be untouched.
    new_nb05 = [_sample_record("nb05-q01"), _sample_record("nb05-q02"),
                _sample_record("nb05-q03")]
    qbank.upsert_slice(5, new_nb05)

    after = qbank.load()
    ids = sorted(q["id"] for q in after["questions"])
    assert ids == ["nb05-q01", "nb05-q02", "nb05-q03", "nb07-q01"]


def test_upsert_slice_is_idempotent_under_repeat(tmp_qbank: Path):
    records = [_sample_record("nb03-q01"), _sample_record("nb03-q02")]

    qbank.upsert_slice(3, records)
    first = copy.deepcopy(qbank.load())

    # Re-upserting the same records must produce the same bank.
    qbank.upsert_slice(3, records)
    second = qbank.load()

    assert first == second


def test_upsert_slice_rejects_foreign_ids(tmp_qbank: Path):
    with pytest.raises(ValueError, match="outside the nb05 slice"):
        qbank.upsert_slice(5, [_sample_record("nb07-q01")])


def test_upsert_slice_rejects_duplicate_ids(tmp_qbank: Path):
    records = [_sample_record("nb02-q01"), _sample_record("nb02-q01")]
    with pytest.raises(ValueError, match="duplicate ids"):
        qbank.upsert_slice(2, records)


# ---------------------------------------------------------------------------
# delete_slice
# ---------------------------------------------------------------------------


def test_delete_slice_removes_only_target_slice(tmp_qbank: Path):
    seed = {
        "questions": [
            _sample_record("nb04-q01"),
            _sample_record("nb04-q02"),
            _sample_record("nb09-q01"),
        ]
    }
    qbank.save(seed)

    qbank.delete_slice(4)
    remaining = qbank.load()
    ids = [q["id"] for q in remaining["questions"]]
    assert ids == ["nb09-q01"]


def test_delete_slice_is_noop_when_slice_is_empty(tmp_qbank: Path):
    qbank.save({"questions": [_sample_record("nb09-q01")]})
    qbank.delete_slice(4)  # nb04 slice doesn't exist
    assert qbank.load() == {"questions": [_sample_record("nb09-q01")]}


# ---------------------------------------------------------------------------
# filter_by_role
# ---------------------------------------------------------------------------


def test_filter_by_role_returns_matching_records(tmp_qbank: Path):
    records = [
        _sample_record("nb01-q01", roles=["mle"]),
        _sample_record("nb01-q02", roles=["research_engineer"]),
        _sample_record("nb01-q03", roles=["mle", "systems_engineer"]),
    ]
    qbank.save({"questions": records})

    mle_records = qbank.filter_by_role("mle")
    mle_ids = sorted(r["id"] for r in mle_records)
    assert mle_ids == ["nb01-q01", "nb01-q03"]

    re_records = qbank.filter_by_role("research_engineer")
    assert [r["id"] for r in re_records] == ["nb01-q02"]

    sys_records = qbank.filter_by_role("systems_engineer")
    assert [r["id"] for r in sys_records] == ["nb01-q03"]


def test_filter_by_role_rejects_unknown_role(tmp_qbank: Path):
    with pytest.raises(ValueError, match="unknown role"):
        qbank.filter_by_role("data_engineer")


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


def test_validate_schema_accepts_a_good_record():
    # Should not raise.
    qbank.validate_schema(_sample_record("nb05-q03"))


def test_validate_schema_rejects_bad_id_pattern():
    bad = _sample_record("nb05-q03")
    bad["id"] = "nb5-q3"  # not zero-padded
    with pytest.raises(ValueError, match="pattern"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_bad_difficulty():
    bad = _sample_record("nb05-q03")
    bad["difficulty"] = "hard"
    with pytest.raises(ValueError, match="difficulty"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_empty_roles():
    bad = _sample_record("nb05-q03")
    bad["roles"] = []
    with pytest.raises(ValueError, match="roles"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_unknown_role():
    bad = _sample_record("nb05-q03")
    bad["roles"] = ["mle", "data_engineer"]
    with pytest.raises(ValueError, match="roles"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_too_few_answer_key_points():
    bad = _sample_record("nb05-q03")
    bad["answer_key_points"] = ["only", "two"]
    with pytest.raises(ValueError, match="answer_key_points"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_too_many_answer_key_points():
    bad = _sample_record("nb05-q03")
    bad["answer_key_points"] = [f"point {i}" for i in range(8)]
    with pytest.raises(ValueError, match="answer_key_points"):
        qbank.validate_schema(bad)


def test_validate_schema_rejects_missing_required_field():
    bad = _sample_record("nb05-q03")
    del bad["trap"]
    with pytest.raises(ValueError, match="missing required fields"):
        qbank.validate_schema(bad)
