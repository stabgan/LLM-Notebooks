"""Verification harness for interview-grade notebook transforms.

Runs three checks on a target notebook:

1. **Execution** — ``jupyter nbconvert --to notebook --execute --inplace``
   (Requirement 7.1, 7.2, 21.1). On failure the harness extracts the
   failing cell index from the exception output and prints it.
2. **Test suite** — ``.venv/bin/python -m pytest tests/ -v --no-header -q``
   and asserts at least 34 passing tests (Requirement 8.1, 8.2).
3. **Q-bank bijection** — every ``### 🎯 Interview Question nb{NN}-q{MM}``
   cell in the notebook must correspond to exactly one record in
   ``question_bank.json`` whose id matches, and vice versa (Requirement
   3.7, 3.8, 21.2).

On any failure the harness exits non-zero with an actionable message.

If an exception contains "out of memory" or "OOM", the harness attempts to
print MLX peak memory via ``mlx.core.metal.get_peak_memory()`` before
exiting (Requirement 21.3).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §C5
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§7, 8, 21
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# scripts/transform/verify.py -> parents[2] is the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QBANK_PATH = (
    _REPO_ROOT
    / ".kiro"
    / "specs"
    / "interview-grade-notebooks"
    / "question_bank.json"
)
_VENV_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
_MIN_PASSING = 34

# Interview question header pattern; matches cells like:
#   ### 🎯 Interview Question nb05-q03  ·  [core]  ·  mle, research_engineer
_QID_RE = re.compile(r"###\s+🎯\s+Interview Question\s+(nb\d{2}-q\d{2})")
_NBCONVERT_CELL_IDX_RE = re.compile(r"CellExecutionError.*?cell\s*(\d+)", re.IGNORECASE)
_NBCONVERT_EXEC_IDX_RE = re.compile(r"failed executing.*?cell\s*(\d+)", re.IGNORECASE)
_INPUT_LINE_RE = re.compile(r"Input In \[(\d+)\]|Cell In\[(\d+)\]")


# ---------------------------------------------------------------------------
# Helpers — pure, unit-testable
# ---------------------------------------------------------------------------


def _locate_notebook(nb_num: int) -> Path:
    """Return the repo-root notebook matching ``{nb_num:02d}_*.ipynb``.

    Raises ``FileNotFoundError`` if zero or multiple candidates match.
    """
    if not isinstance(nb_num, int) or not (0 <= nb_num <= 99):
        raise ValueError(f"nb_num must be an int in 0..99, got {nb_num!r}")
    pattern = f"{nb_num:02d}_*.ipynb"
    matches = sorted(_REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No notebook matched pattern {pattern!r} at repo root"
        )
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise FileNotFoundError(
            f"Multiple notebooks matched pattern {pattern!r}: {names}"
        )
    return matches[0]


def _extract_notebook_qids(notebook_dict: dict) -> list[str]:
    """Return a sorted list of unique qids appearing in 🎯 markdown cells.

    Scans every ``markdown`` cell in ``notebook_dict['cells']`` for a line
    beginning with ``### 🎯 Interview Question nb{NN}-q{MM}`` and captures
    the qid. Duplicates are de-duplicated; the result is sorted.

    Pure function — no I/O — for easy unit testing.
    """
    qids: set[str] = set()
    for cell in notebook_dict.get("cells", []) or []:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for match in _QID_RE.finditer(src):
            qids.add(match.group(1))
    return sorted(qids)


def _extract_qbank_ids(qbank_dict: dict, nb_num: int) -> list[str]:
    """Return sorted qbank ids whose id starts with ``nb{NN}-q``.

    Pure function for easy unit testing.
    """
    prefix = f"nb{nb_num:02d}-q"
    ids = [
        q.get("id", "")
        for q in qbank_dict.get("questions", []) or []
        if isinstance(q.get("id"), str) and q["id"].startswith(prefix)
    ]
    return sorted(ids)


def _check_bijection(
    nb_qids: Iterable[str], qbank_ids: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return ``(missing_in_qbank, missing_in_notebook)`` as sorted lists.

    * ``missing_in_qbank`` — qids present in the notebook but not in the
      question bank slice.
    * ``missing_in_notebook`` — qids present in the question bank slice but
      not in the notebook.

    The bijection holds iff both lists are empty.
    """
    nb_set = set(nb_qids)
    qbank_set = set(qbank_ids)
    missing_in_qbank = sorted(nb_set - qbank_set)
    missing_in_notebook = sorted(qbank_set - nb_set)
    return missing_in_qbank, missing_in_notebook


def _parse_pytest_passing(output: str) -> int | None:
    """Return the number of passing tests parsed from pytest's summary line.

    Pytest summaries look like::

        ==== 34 passed, 2 skipped in 3.21s ====
        ==== 1 failed, 33 passed in 3.21s ====
        ==== 34 passed in 2.00s ====

    Returns ``None`` if no "N passed" token is found.
    """
    match = re.search(r"(\d+)\s+passed", output)
    if match:
        return int(match.group(1))
    return None


def _extract_failing_cell_index(stdout: str, stderr: str) -> int | None:
    """Best-effort extraction of the failing cell index from nbconvert output."""
    blob = f"{stdout}\n{stderr}"
    for pattern in (_NBCONVERT_CELL_IDX_RE, _NBCONVERT_EXEC_IDX_RE):
        m = pattern.search(blob)
        if m:
            return int(m.group(1))
    m = _INPUT_LINE_RE.search(blob)
    if m:
        grp = m.group(1) or m.group(2)
        if grp is not None:
            return int(grp)
    return None


def _maybe_print_peak_memory(message: str) -> None:
    """If ``message`` hints at OOM, try to print MLX peak memory."""
    low = message.lower()
    if "out of memory" not in low and "oom" not in low:
        return
    try:  # MLX may not be importable in all environments
        import mlx.core as mx  # type: ignore[import-not-found]

        try:
            peak = mx.metal.get_peak_memory()
        except Exception:  # pragma: no cover - platform-specific
            peak = None
        if peak is not None:
            print(f"[verify] MLX peak memory at OOM: {peak} bytes", flush=True)
    except ImportError:  # pragma: no cover - MLX not installed
        pass


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------


def _run_nbconvert(notebook: Path) -> tuple[int, str, str]:
    """Execute the notebook in-place via nbconvert. Returns (rc, stdout, stderr)."""
    cmd = [
        str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT)
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_pytest() -> tuple[int, str, str]:
    """Run the project's pytest suite. Returns (rc, stdout, stderr)."""
    cmd = [
        str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--no-header",
        "-q",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT)
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_notebook(nb_num: int, *, skip_execute: bool = False) -> int:
    """Run the three-stage verification pipeline against one notebook.

    Args:
        nb_num: integer 0..99 — the notebook number.
        skip_execute: if True, skip the nbconvert --execute step (useful for
            fast bijection + pytest checks before notebooks have been
            transformed).

    Returns:
        0 on full success; 1 on any failure. Prints actionable diagnostics
        to stdout before returning non-zero.
    """
    try:
        notebook = _locate_notebook(nb_num)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[verify nb{nb_num:02d}] locate failed: {exc}", flush=True)
        return 1

    print(f"[verify nb{nb_num:02d}] target: {notebook.name}", flush=True)

    # --- Stage 1: execute notebook end-to-end ---------------------------------
    if not skip_execute:
        print(f"[verify nb{nb_num:02d}] executing via nbconvert...", flush=True)
        try:
            rc, stdout, stderr = _run_nbconvert(notebook)
        except Exception as exc:  # pragma: no cover - subprocess launch failure
            print(f"[verify nb{nb_num:02d}] nbconvert launch failed: {exc}", flush=True)
            _maybe_print_peak_memory(str(exc))
            return 1
        if rc != 0:
            cell_idx = _extract_failing_cell_index(stdout, stderr)
            if cell_idx is not None:
                print(
                    f"[verify nb{nb_num:02d}] nbconvert failed at cell index {cell_idx}",
                    flush=True,
                )
            else:
                print(
                    f"[verify nb{nb_num:02d}] nbconvert failed (cell index not parsed)",
                    flush=True,
                )
            # Echo the tail of stderr so operators can diagnose.
            tail = "\n".join((stderr or stdout).splitlines()[-30:])
            if tail:
                print(tail, flush=True)
            _maybe_print_peak_memory(f"{stdout}\n{stderr}")
            return 1
    else:
        print(f"[verify nb{nb_num:02d}] skipping nbconvert (--skip-execute)", flush=True)

    # --- Stage 2: pytest ------------------------------------------------------
    print(f"[verify nb{nb_num:02d}] running pytest...", flush=True)
    rc, stdout, stderr = _run_pytest()
    passing = _parse_pytest_passing(stdout + "\n" + stderr)
    if rc != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-20:])
        print(
            f"[verify nb{nb_num:02d}] pytest failed "
            f"(rc={rc}, parsed passing={passing})",
            flush=True,
        )
        if tail:
            print(tail, flush=True)
        return 1
    if passing is None or passing < _MIN_PASSING:
        print(
            f"[verify nb{nb_num:02d}] pytest reported {passing} passing "
            f"(< required {_MIN_PASSING})",
            flush=True,
        )
        return 1
    print(f"[verify nb{nb_num:02d}] pytest OK ({passing} passing)", flush=True)

    # --- Stage 3: Q-bank bijection -------------------------------------------
    print(f"[verify nb{nb_num:02d}] checking Q-bank bijection...", flush=True)
    try:
        with notebook.open("r", encoding="utf-8") as fh:
            nb_dict = json.load(fh)
    except Exception as exc:
        print(f"[verify nb{nb_num:02d}] failed to parse notebook JSON: {exc}", flush=True)
        return 1

    if _QBANK_PATH.exists():
        try:
            with _QBANK_PATH.open("r", encoding="utf-8") as fh:
                qbank_dict = json.load(fh)
        except Exception as exc:
            print(
                f"[verify nb{nb_num:02d}] failed to parse {_QBANK_PATH}: {exc}",
                flush=True,
            )
            return 1
    else:
        qbank_dict = {"questions": []}

    nb_qids = _extract_notebook_qids(nb_dict)
    qbank_ids = _extract_qbank_ids(qbank_dict, nb_num)
    missing_in_qbank, missing_in_notebook = _check_bijection(nb_qids, qbank_ids)

    if missing_in_qbank or missing_in_notebook:
        print(
            f"[verify nb{nb_num:02d}] bijection FAILED: "
            f"{len(missing_in_qbank)} notebook qid(s) missing from qbank, "
            f"{len(missing_in_notebook)} qbank id(s) missing from notebook",
            flush=True,
        )
        if missing_in_qbank:
            print(
                f"  missing_in_qbank:   {missing_in_qbank}", flush=True
            )
        if missing_in_notebook:
            print(
                f"  missing_in_notebook: {missing_in_notebook}", flush=True
            )
        return 1
    print(
        f"[verify nb{nb_num:02d}] bijection OK "
        f"({len(nb_qids)} qid(s) matched)",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_notebooks_arg(spec: str) -> list[int]:
    """Parse a ``--notebooks`` spec into a sorted list of notebook numbers.

    Accepts ranges like ``00-09`` and comma-separated lists like ``00,05,10``.
    Mixing is supported: ``00-04,09,15-17`` yields
    ``[0, 1, 2, 3, 4, 9, 15, 16, 17]``.
    """
    result: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid range {token!r} (lo > hi)")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(token))
    return sorted(result)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify",
        description="Verification harness for interview-grade notebooks.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notebook",
        type=int,
        help="Verify a single notebook (integer, e.g. 5).",
    )
    group.add_argument(
        "--notebooks",
        type=str,
        help="Range (e.g. 00-09) or comma list (e.g. 00,05,10), or mix.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Verify all 20 notebooks (00..19).",
    )
    p.add_argument(
        "--skip-execute",
        action="store_true",
        help="Skip the nbconvert --execute step (fast bijection + pytest).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.all:
        nbs = list(range(0, 20))
    elif args.notebooks is not None:
        nbs = _parse_notebooks_arg(args.notebooks)
    else:
        nbs = [int(args.notebook)]

    print(f"[verify] nb set: {nbs}  skip_execute={args.skip_execute}", flush=True)
    failures: list[int] = []
    for nb in nbs:
        rc = verify_notebook(nb, skip_execute=args.skip_execute)
        if rc != 0:
            failures.append(nb)

    if failures:
        print(
            f"[verify] FAILED for {len(failures)}/{len(nbs)} notebook(s): "
            f"{[f'{n:02d}' for n in failures]}",
            flush=True,
        )
        return 1
    print(f"[verify] OK — {len(nbs)} notebook(s) verified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
