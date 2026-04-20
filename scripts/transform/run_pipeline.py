"""End-to-end pipeline runner for the interview-grade notebook transforms.

For each requested notebook the runner:

1. Dynamically imports ``scripts.transform.nb{NN:02d}`` and calls its
   ``transform()`` entry point (per-notebook additions).
2. Runs the verification harness (``verify.verify_notebook(nb_num,
   skip_execute=...)``) — nbconvert execution, pytest, and Q-bank
   bijection checks (Requirements 7, 8, 21).
3. Optionally commits and pushes via ``commit.commit_and_push`` (Requirement
   16).

Parallelism is provided by ``concurrent.futures.ProcessPoolExecutor`` with
``max_workers = min(args.parallel, 10)`` (Requirement 12.1). Single-notebook
mode runs sequentially in the current process.

**Inline-Python guard:** The pipeline refuses any invocation whose argv
contains ``-c`` so there is no way to inject ad-hoc Python through the CLI
(Requirement 11.3).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-6
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§11, 12, 16, 21.4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import sys
import traceback
from pathlib import Path
from typing import Sequence

# Allow direct script invocation (``python scripts/transform/run_pipeline.py``)
# as well as module invocation (``python -m scripts.transform.run_pipeline``).
# When invoked directly, ``scripts`` is not yet on ``sys.path`` because the
# entry script's directory is what ends up there; prepend the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.transform import commit as commit_mod  # noqa: E402
from scripts.transform import verify as verify_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NB_RANGE: tuple[int, ...] = tuple(range(0, 20))  # 00..19
_MAX_PARALLEL = 10
_TRANSFORM_STAGE = "interview-grade transform"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------


def _reject_inline_python(argv: Sequence[str]) -> None:
    """Raise ``RuntimeError`` if ``argv`` contains a bare ``-c`` token.

    Requirement 11.3: the pipeline must not accept inline Python code
    passed via the ``-c`` flag. We reject at the start of ``main`` so the
    error message is unambiguous.
    """
    for token in argv:
        if token == "-c":
            raise RuntimeError(
                "inline python (-c) is forbidden; use a script under "
                "scripts/transform/"
            )


def _resolve_notebooks(*, all: bool, notebook: int | None) -> list[int]:
    """Return the sorted list of notebook numbers to process.

    ``all=True`` → ``[0..19]``. Otherwise ``notebook`` must be an int in
    0..19 and a singleton list is returned. The helper is intentionally
    keyword-only so it's trivially covered by unit tests.
    """
    if all:
        return list(_NB_RANGE)
    if notebook is None:
        raise ValueError("notebook must be provided when all=False")
    if not isinstance(notebook, int) or not (0 <= notebook <= 99):
        raise ValueError(
            f"notebook must be an int in 0..99, got {notebook!r}"
        )
    return [int(notebook)]


def _clamp_parallel(parallel: int) -> int:
    """Clamp ``parallel`` to the range ``[1, 10]``."""
    if not isinstance(parallel, int):
        raise ValueError(f"parallel must be int, got {parallel!r}")
    if parallel < 1:
        return 1
    return min(parallel, _MAX_PARALLEL)


# ---------------------------------------------------------------------------
# Per-notebook pipeline
# ---------------------------------------------------------------------------


def _run_one(
    nb_num: int,
    *,
    push: bool,
    skip_execute: bool,
    skip_review: bool,  # accepted for API parity; hook is a no-op for now
) -> int:
    """Run the full pipeline for one notebook. Returns 0 on success.

    Steps:
        1. Import ``scripts.transform.nb{NN:02d}`` and call ``.transform()``.
           If the module doesn't exist yet, print a warning and return
           non-zero so callers can see unfinished work without erroring.
        2. Run ``verify.verify_notebook(nb_num, skip_execute=...)``.
        3. Optionally ``commit.commit_and_push(nb_num, ..., push=push)``.
    """
    tag = f"[pipeline nb{nb_num:02d}]"
    module_name = f"scripts.transform.nb{nb_num:02d}"

    # Step 1 — transform ---------------------------------------------------
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        # Distinguish a missing module from an import-time failure inside it.
        msg = str(exc)
        if module_name.split(".")[-1] in msg or "No module named" in msg:
            print(
                f"{tag} transform module not found: {module_name} "
                f"(skipping; create scripts/transform/nb{nb_num:02d}.py)",
                flush=True,
            )
            return 1
        print(
            f"{tag} import of {module_name} failed: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return 1

    transform_fn = getattr(mod, "transform", None)
    if not callable(transform_fn):
        print(
            f"{tag} {module_name} has no callable 'transform()' entry point",
            flush=True,
        )
        return 1

    print(f"{tag} running {module_name}.transform()...", flush=True)
    try:
        transform_fn()
    except Exception as exc:  # surface the exception; don't swallow it
        print(f"{tag} transform raised: {exc}", flush=True)
        traceback.print_exc()
        return 1

    # Step 2 — verify ------------------------------------------------------
    rc = verify_mod.verify_notebook(nb_num, skip_execute=skip_execute)
    if rc != 0:
        print(f"{tag} verification failed (rc={rc}); skipping commit/push", flush=True)
        return rc

    # Step 3 — commit (+ optional push) -----------------------------------
    try:
        sha = commit_mod.commit_and_push(
            nb_num, _TRANSFORM_STAGE, push=push
        )
        print(f"{tag} commit sha: {sha}", flush=True)
    except Exception as exc:  # pragma: no cover - unusual git failure
        print(f"{tag} commit_and_push failed: {exc}", flush=True)
        return 1

    if skip_review:
        print(f"{tag} --skip-review set; pedagogy review not invoked", flush=True)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "End-to-end runner for interview-grade notebook transforms: "
            "transform → verify → commit → push."
        ),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notebook",
        type=int,
        help="Run the pipeline for a single notebook (e.g. 5).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run the pipeline for all 20 notebooks (00..19).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=_MAX_PARALLEL,
        help=f"Max concurrent notebooks (1..{_MAX_PARALLEL}, default {_MAX_PARALLEL}).",
    )
    p.add_argument(
        "--push",
        action="store_true",
        default=False,
        help="Push commits to origin/main after each notebook succeeds.",
    )
    p.add_argument(
        "--skip-review",
        action="store_true",
        default=False,
        help="Skip the pedagogy-review sub-agent invocation.",
    )
    p.add_argument(
        "--skip-execute",
        action="store_true",
        default=False,
        help="Skip the `nbconvert --execute` step in verify (faster).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Hard guard: reject inline Python invocations on the CLI.
    assert "-c" not in sys.argv, (
        "inline python (-c) is forbidden; use a script under scripts/transform/"
    )
    _reject_inline_python(sys.argv)

    args = _build_parser().parse_args(argv)

    nbs = _resolve_notebooks(all=args.all, notebook=args.notebook)
    parallel = _clamp_parallel(args.parallel)

    print(
        f"[pipeline] nb set: {nbs}  parallel={parallel}  push={args.push}  "
        f"skip_review={args.skip_review}  skip_execute={args.skip_execute}",
        flush=True,
    )

    # Sequential when running one notebook — simpler + better stack traces.
    if len(nbs) == 1:
        rc = _run_one(
            nbs[0],
            push=args.push,
            skip_execute=args.skip_execute,
            skip_review=args.skip_review,
        )
        return 0 if rc == 0 else rc

    # Parallel execution for multiple notebooks.
    failures: list[int] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as pool:
        future_to_nb = {
            pool.submit(
                _run_one,
                nb,
                push=args.push,
                skip_execute=args.skip_execute,
                skip_review=args.skip_review,
            ): nb
            for nb in nbs
        }
        for fut in concurrent.futures.as_completed(future_to_nb):
            nb = future_to_nb[fut]
            try:
                rc = fut.result()
            except Exception as exc:  # pragma: no cover - worker crash
                print(
                    f"[pipeline nb{nb:02d}] worker raised: {exc}", flush=True
                )
                rc = 1
            if rc != 0:
                failures.append(nb)

    if failures:
        print(
            f"[pipeline] FAILED for {len(failures)}/{len(nbs)} notebook(s): "
            f"{[f'{n:02d}' for n in sorted(failures)]}",
            flush=True,
        )
        return 1
    print(f"[pipeline] OK — {len(nbs)} notebook(s) completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
