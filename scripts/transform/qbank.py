"""Question-bank infrastructure for the interview-grade notebook transform.

This module is the single point of contact for ``question_bank.json``.
Concurrent writers (up to the pipeline's parallelism cap of 10) coordinate
through a `filelock`-backed `.lock` file next to the bank. Each transform
owns only its own ``nb{NN}-q*`` slice and must not touch records belonging
to other notebooks.

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §C3, §D1
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §3 (schema),
      §12.2-12.4 (concurrency), §17.1-17.2 (role filter).

Public API:
    QBANK_PATH, LOCK_PATH           — canonical absolute paths
    load()                          — read the bank
    save(data)                      — atomic write (tmpfile + rename)
    upsert_slice(nb_num, records)   — replace one notebook's slice
    delete_slice(nb_num)            — drop one notebook's slice
    filter_by_role(role)            — role-filtered view
    validate_schema(record)         — raises ValueError on any violation

Locking:
    Prefers the ``filelock`` package. If unavailable, falls back to a
    macOS-compatible ``fcntl.flock`` advisory lock on the same ``.lock``
    file. Both strategies use the same file so mixed-strategy writers
    coordinate correctly on a single machine.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# scripts/transform/qbank.py  ->  parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

QBANK_PATH: Path = (
    _REPO_ROOT
    / ".kiro"
    / "specs"
    / "interview-grade-notebooks"
    / "question_bank.json"
)
LOCK_PATH: Path = QBANK_PATH.with_suffix(".lock")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^nb\d{2}-q\d{2}$")
_VALID_DIFFICULTIES: frozenset[str] = frozenset(
    {"warmup", "core", "stretch", "research"}
)
_VALID_ROLES: frozenset[str] = frozenset(
    {"mle", "research_engineer", "systems_engineer"}
)
_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "notebook",
    "section",
    "difficulty",
    "roles",
    "topic_tags",
    "question",
    "answer_key_points",
    "worked_solution_cell_id",
    "trap",
    "references",
    "added_in",
)


def validate_schema(record: dict) -> None:
    """Raise ``ValueError`` if ``record`` violates the Question_Bank schema.

    Enforces Requirement 3.2-3.6:
      * all required fields present,
      * ``id`` matches ``^nb\\d{2}-q\\d{2}$``,
      * ``difficulty`` in {warmup, core, stretch, research},
      * ``roles`` is a non-empty subset of the three known roles,
      * ``answer_key_points`` is a list with 3..7 items inclusive.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")

    # Required fields
    missing = [f for f in _REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    # id
    qid = record["id"]
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        raise ValueError(
            f"id must match pattern '^nb\\d{{2}}-q\\d{{2}}$', got {qid!r}"
        )

    # difficulty
    diff = record["difficulty"]
    if diff not in _VALID_DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {sorted(_VALID_DIFFICULTIES)}, "
            f"got {diff!r}"
        )

    # roles: non-empty subset of known roles
    roles = record["roles"]
    if not isinstance(roles, list) or not roles:
        raise ValueError(
            f"roles must be a non-empty list, got {roles!r}"
        )
    bad_roles = [r for r in roles if r not in _VALID_ROLES]
    if bad_roles:
        raise ValueError(
            f"roles contains unknown values {bad_roles!r}; "
            f"valid: {sorted(_VALID_ROLES)}"
        )

    # answer_key_points: 3..7 items
    akp = record["answer_key_points"]
    if not isinstance(akp, list):
        raise ValueError(
            f"answer_key_points must be a list, got {type(akp).__name__}"
        )
    if not (3 <= len(akp) <= 7):
        raise ValueError(
            f"answer_key_points must have 3..7 items, got {len(akp)}"
        )


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

try:  # prefer `filelock` if installed
    from filelock import FileLock as _FileLock  # type: ignore[import-not-found]

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - exercised when filelock is missing
    _FileLock = None  # type: ignore[assignment]
    _HAS_FILELOCK = False


@contextlib.contextmanager
def _qbank_lock(timeout: float = 30.0) -> Iterator[None]:
    """Acquire an exclusive lock on ``LOCK_PATH`` for the duration of the block.

    Uses ``filelock.FileLock`` when available; otherwise falls back to
    ``fcntl.flock`` (macOS + Linux compatible, advisory locking).
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_FILELOCK:
        lock = _FileLock(str(LOCK_PATH), timeout=timeout)
        with lock:
            yield
        return

    # fcntl fallback (macOS-compatible advisory lock)
    import fcntl  # local import so Windows users can at least import the module

    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load() -> dict[str, Any]:
    """Read the Question_Bank and return its parsed JSON dict.

    If the file does not yet exist, returns ``{"questions": []}``.
    Reads are not locked; callers that need a consistent view alongside
    their write should hold the lock themselves.
    """
    if not QBANK_PATH.exists():
        return {"questions": []}
    with QBANK_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(
            f"{QBANK_PATH} is malformed: expected object with 'questions' key"
        )
    return data


def save(data: dict[str, Any]) -> None:
    """Atomically write ``data`` to ``QBANK_PATH`` under the file lock.

    Uses a tmpfile + ``os.replace`` so readers never see a half-written
    file. Takes the qbank lock for the full duration of the write
    (Requirement 12.2).
    """
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("data must be a dict with a 'questions' key")

    QBANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _qbank_lock():
        # Write to a sibling tmpfile then atomically rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".question_bank.",
            suffix=".tmp",
            dir=str(QBANK_PATH.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_name, QBANK_PATH)
        except Exception:
            # Best-effort tmpfile cleanup on failure.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise


# ---------------------------------------------------------------------------
# Slice operations
# ---------------------------------------------------------------------------

def _slice_prefix(nb_num: int) -> str:
    """Return the id prefix that identifies one notebook's slice."""
    if not isinstance(nb_num, int) or not (0 <= nb_num <= 99):
        raise ValueError(f"nb_num must be an int in 0..99, got {nb_num!r}")
    return f"nb{nb_num:02d}-q"


def upsert_slice(nb_num: int, records: list[dict]) -> None:
    """Replace all ``nb{NN}-q*`` records with ``records``; other slices untouched.

    Every record in ``records`` is validated before any write happens, and
    every id must belong to this notebook's slice (Requirement 12.3). The
    read + merge + write runs under the qbank lock so concurrent upserts
    targeting other notebooks never race.
    """
    prefix = _slice_prefix(nb_num)

    # Pre-validate inputs outside the lock — cheap and fails fast.
    for record in records:
        validate_schema(record)
        if not record["id"].startswith(prefix):
            raise ValueError(
                f"record id {record['id']!r} is outside the nb{nb_num:02d} "
                f"slice (expected prefix {prefix!r})"
            )

    # Reject duplicate ids inside the incoming slice.
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"duplicate ids in records: {dupes}")

    with _qbank_lock():
        data = load()
        other = [
            q for q in data.get("questions", [])
            if not q.get("id", "").startswith(prefix)
        ]
        data["questions"] = other + list(records)
        # Inlined save body so we don't re-enter the lock.
        _atomic_write(data)


def delete_slice(nb_num: int) -> None:
    """Remove every record whose id starts with ``nb{NN}-q``.

    No-op if the slice is already empty.
    """
    prefix = _slice_prefix(nb_num)
    with _qbank_lock():
        data = load()
        data["questions"] = [
            q for q in data.get("questions", [])
            if not q.get("id", "").startswith(prefix)
        ]
        _atomic_write(data)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def filter_by_role(role: str) -> list[dict]:
    """Return records whose ``roles`` field contains ``role``.

    Validates the role against the known set before filtering to catch
    typos early (Requirement 17.2).
    """
    if role not in _VALID_ROLES:
        raise ValueError(
            f"unknown role {role!r}; valid: {sorted(_VALID_ROLES)}"
        )
    data = load()
    return [q for q in data.get("questions", []) if role in (q.get("roles") or [])]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write(data: dict[str, Any]) -> None:
    """Write ``data`` atomically without taking the lock.

    Used internally by functions that already hold the lock (``upsert_slice``,
    ``delete_slice``) to avoid double-acquire on non-reentrant locks.
    """
    QBANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".question_bank.",
        suffix=".tmp",
        dir=str(QBANK_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, QBANK_PATH)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
