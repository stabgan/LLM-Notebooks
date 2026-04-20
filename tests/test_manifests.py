"""Smoke tests for `scripts.transform.manifests`.

Covers:
  - load_manifest returns a validated NotebookManifest for every nb_num in 0..19
  - list_all_manifests returns 20 manifests in order
  - Filename <-> nb_num consistency (derived from the loader)
  - preserve_sections has >= 3 entries; anchor_after_headings has >= 2
  - Score invariants: 0.0 <= pedagogy_score_before <= 10.0, target_score_after >= 8.0
  - Count minimums from requirements.md §1: questions>=6, whiteboard>=2,
    complexity>=2, production>=1, frontier>=1, debugging>=1
  - Validation rejects malformed payloads (missing fields, bad scores, etc.)

Run with:
    .venv/bin/python -m pytest tests/test_manifests.py -v --no-header
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.transform.manifests import (
    MANIFESTS_DIR,
    NotebookManifest,
    list_all_manifests,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Load-every-manifest smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nb_num", list(range(20)))
def test_load_manifest_succeeds_for_every_notebook(nb_num: int) -> None:
    m = load_manifest(nb_num)
    assert isinstance(m, NotebookManifest)
    assert m.nb_num == nb_num


def test_list_all_manifests_returns_twenty_in_order() -> None:
    all_manifests = list_all_manifests()
    assert len(all_manifests) == 20
    assert [m.nb_num for m in all_manifests] == list(range(20))


# ---------------------------------------------------------------------------
# Per-manifest invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nb_num", list(range(20)))
def test_filename_matches_nb_num(nb_num: int) -> None:
    m = load_manifest(nb_num)
    assert m.notebook.endswith(".ipynb")
    assert m.notebook.startswith(f"{nb_num:02d}_"), (
        f"filename {m.notebook!r} should start with {nb_num:02d}_"
    )


@pytest.mark.parametrize("nb_num", list(range(20)))
def test_preserve_and_anchor_sizes(nb_num: int) -> None:
    m = load_manifest(nb_num)
    assert len(m.preserve_sections) >= 3, (
        f"nb{nb_num:02d}: preserve_sections must have >=3 entries"
    )
    assert len(m.anchor_after_headings) >= 2, (
        f"nb{nb_num:02d}: anchor_after_headings must have >=2 entries"
    )
    # Every entry is a non-empty string
    assert all(isinstance(s, str) and s.strip() for s in m.preserve_sections)
    assert all(isinstance(s, str) and s.strip() for s in m.anchor_after_headings)


@pytest.mark.parametrize("nb_num", list(range(20)))
def test_score_invariants(nb_num: int) -> None:
    m = load_manifest(nb_num)
    assert 0.0 <= m.pedagogy_score_before <= 10.0
    assert m.target_score_after >= 8.0
    assert m.target_score_after <= 10.0
    assert m.target_score_after >= m.pedagogy_score_before


@pytest.mark.parametrize("nb_num", list(range(20)))
def test_count_minimums(nb_num: int) -> None:
    """Counts per stratum must meet requirements.md §1 minimums."""
    m = load_manifest(nb_num)
    assert m.questions_to_add >= 6
    assert m.whiteboard_challenges_to_add >= 2
    assert m.complexity_cells_to_add >= 2
    assert m.production_cells_to_add >= 1
    assert m.frontier_cells_to_add >= 1
    assert m.debugging_cells_to_add >= 1


# ---------------------------------------------------------------------------
# Directory + file layout
# ---------------------------------------------------------------------------


def test_manifests_dir_contains_exactly_20_json_files() -> None:
    files = sorted(p.name for p in MANIFESTS_DIR.glob("nb*.json"))
    expected = [f"nb{n:02d}.json" for n in range(20)]
    assert files == expected


# ---------------------------------------------------------------------------
# Validation failure modes
# ---------------------------------------------------------------------------


def test_load_manifest_rejects_out_of_range_nb_num() -> None:
    with pytest.raises(ValueError):
        load_manifest(20)
    with pytest.raises(ValueError):
        load_manifest(-1)


def test_load_manifest_rejects_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point MANIFESTS_DIR at an empty tmpdir and request a valid nb_num.
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        mod.load_manifest(0)


def _base_payload(nb_num: int = 0) -> dict:
    return {
        "notebook": f"{nb_num:02d}_something.ipynb",
        "nb_num": nb_num,
        "pedagogy_score_before": 6.5,
        "target_score_after": 8.0,
        "questions_to_add": 6,
        "whiteboard_challenges_to_add": 2,
        "complexity_cells_to_add": 2,
        "production_cells_to_add": 1,
        "frontier_cells_to_add": 1,
        "debugging_cells_to_add": 1,
        "preserve_sections": ["A", "B", "C"],
        "anchor_after_headings": ["X", "Y"],
    }


def test_validation_rejects_target_below_interview_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    payload = _base_payload(0)
    payload["target_score_after"] = 7.9
    (tmp_path / "nb00.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target_score_after"):
        mod.load_manifest(0)


def test_validation_rejects_score_before_out_of_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    payload = _base_payload(0)
    payload["pedagogy_score_before"] = 11.0
    (tmp_path / "nb00.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pedagogy_score_before"):
        mod.load_manifest(0)


def test_validation_rejects_count_below_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    payload = _base_payload(0)
    payload["questions_to_add"] = 5  # below minimum of 6
    (tmp_path / "nb00.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="questions_to_add"):
        mod.load_manifest(0)


def test_validation_rejects_preserve_sections_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    payload = _base_payload(0)
    payload["preserve_sections"] = ["Only", "Two"]
    (tmp_path / "nb00.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="preserve_sections"):
        mod.load_manifest(0)


def test_validation_rejects_filename_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.transform import manifests as mod

    monkeypatch.setattr(mod, "MANIFESTS_DIR", tmp_path)
    payload = _base_payload(0)
    payload["notebook"] = "99_wrong.ipynb"  # filename prefix doesn't match nb_num=0
    (tmp_path / "nb00.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nb_num"):
        mod.load_manifest(0)
