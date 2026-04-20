"""Per-notebook transformation manifests.

Each of the 20 notebooks has a JSON manifest under ``scripts/transform/manifests/``
(``nb00.json`` … ``nb19.json``) that encodes how many cells of each stratum to
insert, which sections must stay byte-identical (``preserve_sections``), and
which headings serve as anchors for new interview-layer cells
(``anchor_after_headings``).

The schema comes from ``.kiro/specs/interview-grade-notebooks/design.md`` §D3.
``pydantic`` isn't currently in ``requirements.txt``, so we fall back to a
``dataclass`` with hand-rolled ``__post_init__`` validation — same contract,
zero new dependencies.

Public API:
    NotebookManifest                 — dataclass with validated fields
    load_manifest(nb_num)            — load + validate a single manifest
    list_all_manifests()             — load + validate all 20

Design references:
    - design.md §D3 (NotebookManifest schema)
    - design.md §LLD-4 (per-notebook previous pedagogy scores)
    - requirements.md §6.2, §6.4, §10.3 (narrative preservation + anchoring)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# scripts/transform/manifests.py  ->  parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR: Path = _REPO_ROOT / "scripts" / "transform" / "manifests"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Minimum counts per stratum (design.md §HLD-1 / requirements.md §1).
_MIN_QUESTIONS = 6
_MIN_WHITEBOARD = 2
_MIN_COMPLEXITY = 2
_MIN_PRODUCTION = 1
_MIN_FRONTIER = 1
_MIN_DEBUGGING = 1

_MIN_PRESERVE_SECTIONS = 3
_MIN_ANCHORS = 2


@dataclass
class NotebookManifest:
    """One notebook's transformation plan.

    Field semantics mirror design.md §D3. All counts are lower bounds
    enforced here; individual transform scripts may insert more, but never
    fewer. Validation runs in ``__post_init__`` so every manifest that
    leaves this module is guaranteed well-formed.
    """

    notebook: str
    nb_num: int
    pedagogy_score_before: float
    target_score_after: float
    questions_to_add: int
    whiteboard_challenges_to_add: int
    complexity_cells_to_add: int
    production_cells_to_add: int
    frontier_cells_to_add: int
    debugging_cells_to_add: int
    preserve_sections: list[str] = field(default_factory=list)
    anchor_after_headings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:  # noqa: C901 - linear validation block
        # notebook filename
        if not isinstance(self.notebook, str) or not self.notebook.endswith(".ipynb"):
            raise ValueError(
                f"notebook must be a .ipynb filename, got {self.notebook!r}"
            )

        # nb_num in 0..19
        if not isinstance(self.nb_num, int) or not (0 <= self.nb_num <= 19):
            raise ValueError(
                f"nb_num must be an int in 0..19, got {self.nb_num!r}"
            )

        # Filename / nb_num consistency: filename must start with "{NN}_"
        expected_prefix = f"{self.nb_num:02d}_"
        if not self.notebook.startswith(expected_prefix):
            raise ValueError(
                f"notebook filename {self.notebook!r} does not match "
                f"nb_num={self.nb_num} (expected prefix {expected_prefix!r})"
            )

        # Scores: 0..10 for previous, >=8 for target, target >= previous.
        if not isinstance(self.pedagogy_score_before, (int, float)):
            raise ValueError("pedagogy_score_before must be numeric")
        if not (0.0 <= float(self.pedagogy_score_before) <= 10.0):
            raise ValueError(
                f"pedagogy_score_before must be in [0.0, 10.0], "
                f"got {self.pedagogy_score_before}"
            )
        if not isinstance(self.target_score_after, (int, float)):
            raise ValueError("target_score_after must be numeric")
        if float(self.target_score_after) < 8.0:
            raise ValueError(
                f"target_score_after must be >= 8.0 (interview-grade bar), "
                f"got {self.target_score_after}"
            )
        if float(self.target_score_after) > 10.0:
            raise ValueError(
                f"target_score_after must be <= 10.0, "
                f"got {self.target_score_after}"
            )
        if float(self.target_score_after) < float(self.pedagogy_score_before):
            raise ValueError(
                "target_score_after must be >= pedagogy_score_before "
                f"({self.target_score_after} < {self.pedagogy_score_before})"
            )

        # Counts: minimums from requirements.md §1.
        _check_min("questions_to_add", self.questions_to_add, _MIN_QUESTIONS)
        _check_min(
            "whiteboard_challenges_to_add",
            self.whiteboard_challenges_to_add,
            _MIN_WHITEBOARD,
        )
        _check_min(
            "complexity_cells_to_add",
            self.complexity_cells_to_add,
            _MIN_COMPLEXITY,
        )
        _check_min(
            "production_cells_to_add",
            self.production_cells_to_add,
            _MIN_PRODUCTION,
        )
        _check_min(
            "frontier_cells_to_add", self.frontier_cells_to_add, _MIN_FRONTIER
        )
        _check_min(
            "debugging_cells_to_add",
            self.debugging_cells_to_add,
            _MIN_DEBUGGING,
        )

        # preserve_sections / anchor_after_headings
        _check_heading_list(
            "preserve_sections", self.preserve_sections, _MIN_PRESERVE_SECTIONS
        )
        _check_heading_list(
            "anchor_after_headings", self.anchor_after_headings, _MIN_ANCHORS
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _manifest_path(nb_num: int) -> Path:
    if not isinstance(nb_num, int) or not (0 <= nb_num <= 19):
        raise ValueError(f"nb_num must be an int in 0..19, got {nb_num!r}")
    return MANIFESTS_DIR / f"nb{nb_num:02d}.json"


def load_manifest(nb_num: int) -> NotebookManifest:
    """Load and validate the manifest JSON for ``nb_num`` (0..19)."""
    path = _manifest_path(nb_num)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}")
    return _build_manifest(raw, source=path)


def list_all_manifests() -> list[NotebookManifest]:
    """Return the 20 manifests in order (nb00 … nb19)."""
    return [load_manifest(n) for n in range(20)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_manifest(raw: dict[str, Any], *, source: Path) -> NotebookManifest:
    """Construct a NotebookManifest from a raw JSON dict, surfacing the source path."""
    try:
        return NotebookManifest(
            notebook=raw["notebook"],
            nb_num=raw["nb_num"],
            pedagogy_score_before=float(raw["pedagogy_score_before"]),
            target_score_after=float(raw["target_score_after"]),
            questions_to_add=int(raw["questions_to_add"]),
            whiteboard_challenges_to_add=int(raw["whiteboard_challenges_to_add"]),
            complexity_cells_to_add=int(raw["complexity_cells_to_add"]),
            production_cells_to_add=int(raw["production_cells_to_add"]),
            frontier_cells_to_add=int(raw["frontier_cells_to_add"]),
            debugging_cells_to_add=int(raw["debugging_cells_to_add"]),
            preserve_sections=list(raw.get("preserve_sections", [])),
            anchor_after_headings=list(raw.get("anchor_after_headings", [])),
        )
    except KeyError as exc:
        raise ValueError(f"{source}: missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        # Re-raise with the manifest path so the caller knows which file broke.
        raise ValueError(f"{source}: {exc}") from exc


def _check_min(name: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _check_heading_list(name: str, value: Any, minimum: int) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list, got {type(value).__name__}")
    if len(value) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} entries, got {len(value)}")
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{name}[{i}] must be a non-empty string, got {item!r}"
            )
