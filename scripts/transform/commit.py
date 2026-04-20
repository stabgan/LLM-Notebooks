"""Git commit + push helpers for the interview-grade notebook pipeline.

This module stages the notebook and its Q-bank slice, creates a
conventional-commit message, commits locally, and optionally pushes to
``origin main``. It returns the short (7-char) Git SHA of the resulting
commit so callers can backfill ``added_in`` metadata in the Question_Bank
(Requirement 16.4).

If there is nothing to commit (clean working tree) the function returns the
current ``HEAD`` SHA without erroring — re-runs on an already-committed
notebook are a no-op. If ``git push`` fails we print a warning but do NOT
re-raise: the local commit has still succeeded (Requirement 16.1).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-6
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§16.1-16.4
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# scripts/transform/commit.py -> parents[2] is the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QBANK_PATH = (
    _REPO_ROOT
    / ".kiro"
    / "specs"
    / "interview-grade-notebooks"
    / "question_bank.json"
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------


def _build_message(nb_num: int, stage: str) -> str:
    """Return the conventional-commit message for this notebook + stage.

    Format: ``feat(nb{NN:02d}): {stage}`` (Requirement 16.3). Trims the
    stage of any surrounding whitespace so accidental padding doesn't leak
    into the commit log.
    """
    if not isinstance(nb_num, int) or not (0 <= nb_num <= 99):
        raise ValueError(f"nb_num must be an int in 0..99, got {nb_num!r}")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError(f"stage must be a non-empty string, got {stage!r}")
    return f"feat(nb{nb_num:02d}): {stage.strip()}"


def _notebook_glob_paths(nb_num: int) -> list[Path]:
    """Return paths under repo root matching ``{nb_num:02d}_*.ipynb``."""
    return sorted(_REPO_ROOT.glob(f"{nb_num:02d}_*.ipynb"))


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------


def _run_git(
    *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in the repo root, capturing stdout/stderr as text."""
    return subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=check,
    )


def _current_head_sha(short: bool = True) -> str:
    """Return the current ``HEAD`` SHA; short (7 chars) by default."""
    args = ["rev-parse", "--short=7", "HEAD"] if short else ["rev-parse", "HEAD"]
    proc = _run_git(*args, check=True)
    return proc.stdout.strip()


def _has_staged_changes() -> bool:
    """True iff ``git diff --cached --quiet`` reports staged changes."""
    # ``git diff --cached --quiet`` exits 1 when there are staged changes.
    proc = _run_git("diff", "--cached", "--quiet", check=False)
    return proc.returncode != 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def commit_and_push(nb_num: int, stage: str, *, push: bool = True) -> str:
    """Stage, commit, and optionally push changes for one notebook.

    Stages the target notebook (``{NN:02d}_*.ipynb`` at the repo root) and
    the Question_Bank file, then runs ``git commit -m "feat(nb{NN}): {stage}"``.
    If nothing is staged, the function returns the current ``HEAD`` SHA
    without erroring. If ``push=True`` we also run ``git push origin main``;
    push failures are logged as warnings but do not abort (the local commit
    has still succeeded).

    Args:
        nb_num: notebook number in 0..99.
        stage: short stage description (e.g. ``"interview-grade transform"``).
        push: whether to attempt ``git push origin main`` after commit.

    Returns:
        Short (7-char) Git SHA of the resulting commit, or the current
        ``HEAD`` SHA if there was nothing to commit.
    """
    message = _build_message(nb_num, stage)

    # Stage the notebook(s) matching the number, plus the Q-bank slice.
    to_add: list[str] = []
    for nb_path in _notebook_glob_paths(nb_num):
        to_add.append(str(nb_path.relative_to(_REPO_ROOT)))
    if _QBANK_PATH.exists():
        to_add.append(str(_QBANK_PATH.relative_to(_REPO_ROOT)))

    if to_add:
        # ``git add`` tolerates unchanged paths; it's a no-op on them.
        _run_git("add", "--", *to_add, check=False)

    # If there's nothing staged, skip commit and just return current HEAD.
    if not _has_staged_changes():
        print(
            f"[commit nb{nb_num:02d}] nothing to commit; returning current HEAD",
            flush=True,
        )
        return _current_head_sha(short=True)

    # Commit locally.
    commit_proc = _run_git("commit", "-m", message, check=False)
    if commit_proc.returncode != 0:
        # Shouldn't happen (we verified staged changes), but surface the error.
        err = (commit_proc.stderr or commit_proc.stdout).strip()
        raise RuntimeError(
            f"git commit failed for nb{nb_num:02d}: {err}"
        )

    sha = _current_head_sha(short=True)
    print(
        f"[commit nb{nb_num:02d}] committed {sha}: {message}",
        flush=True,
    )

    # Optionally push.
    if push:
        push_proc = _run_git("push", "origin", "main", check=False)
        if push_proc.returncode != 0:
            tail = (push_proc.stderr or push_proc.stdout).strip()
            print(
                f"[commit nb{nb_num:02d}] WARNING: git push failed "
                f"(commit {sha} is local-only): {tail}",
                flush=True,
            )
        else:
            print(
                f"[commit nb{nb_num:02d}] pushed {sha} to origin/main",
                flush=True,
            )

    return sha
