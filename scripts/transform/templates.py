"""Cell template library for interview-grade notebook transforms.

Each template function is a **pure** function (no I/O, no global state) that
returns an MCP-Jupyter-compatible cell dict::

    {"cell_type": "markdown" | "code", "source": "<string>"}

The cells are emitted verbatim by `mcp_jupyter_editor_ipynb_insert_cell`.

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §C1, §LLD-1
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §20 (emoji rules)

Emoji prefixes (Requirement 20):
    🎯 Interview Question
    🧑‍💻 Whiteboard Challenge
    📐 Complexity Analysis
    🏭 Production Context
    🔭 Frontier Context
    🛠️ Debugging & Failure Modes
    📋 Interview Question Index
"""

from __future__ import annotations

from typing import Literal, Sequence, TypedDict


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

Difficulty = Literal["warmup", "core", "stretch", "research"]
Role = Literal["mle", "research_engineer", "systems_engineer"]


class InterviewQuestion(TypedDict, total=False):
    """Structured record for a single interview question.

    Matches the Question_Bank schema in design §D1. Optional fields
    (``trap``, ``references``, ``worked_solution_cell_id``) may be omitted
    when building an in-notebook cell; required fields are validated by
    `qbank.validate_schema` before being written to ``question_bank.json``.
    """

    id: str                       # e.g. "nb05-q03" — pattern nb{NN}-q{NN}
    notebook: str                 # filename, e.g. "05_self_attention.ipynb"
    section: str                  # markdown heading the Q lives under
    difficulty: Difficulty
    roles: list[Role]
    topic_tags: list[str]
    question: str
    answer_key_points: list[str]  # 3-7 items
    worked_solution_cell_id: str | None
    trap: str | None
    references: list[str]
    added_in: str                 # git sha of the transform commit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _md(source: str) -> dict:
    """Build a markdown cell dict."""
    return {"cell_type": "markdown", "source": source}


def _code(source: str) -> dict:
    """Build a code cell dict."""
    return {"cell_type": "code", "source": source}


def _format_roles(roles: Sequence[Role]) -> str:
    """Render a roles list as a comma-separated string used in headings."""
    return ", ".join(roles)


def _bullets(items: Sequence[str]) -> str:
    """Render a list of strings as a markdown bullet list."""
    return "\n".join(f"- {item}" for item in items)


def _numbered(items: Sequence[str]) -> str:
    """Render a list of strings as a numbered markdown list (1. 2. …)."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


# ---------------------------------------------------------------------------
# A. Interview Question cell (markdown)
# ---------------------------------------------------------------------------

def interview_question_cell(q: InterviewQuestion) -> dict:
    """Markdown cell for an interview question.

    Format matches design §LLD-1:

        ### 🎯 Interview Question {id}  ·  [{difficulty}]  ·  {roles}

        **Q:** {question}

        <details>
        <summary>Key points in a strong answer</summary>

        - {point 1}
        - ...
        </details>

        > ⚠️ **Trap:** {trap}
        >
        > 📚 **References:** {refs joined by ", "}

    The emoji prefix `🎯` is mandatory (Requirement 20.1).
    """
    qid = q["id"]
    difficulty = q["difficulty"]
    roles = _format_roles(q["roles"])
    question = q["question"]
    key_points = q["answer_key_points"]
    trap = q.get("trap")
    refs = q.get("references") or []

    lines: list[str] = [
        f"### 🎯 Interview Question {qid}  ·  [{difficulty}]  ·  {roles}",
        "",
        f"**Q:** {question}",
        "",
        "<details>",
        "<summary>Key points in a strong answer</summary>",
        "",
        _bullets(key_points),
        "</details>",
    ]

    if trap:
        lines += ["", f"> ⚠️ **Trap:** {trap}"]
        if refs:
            lines += [">", f"> 📚 **References:** {', '.join(refs)}"]
    elif refs:
        lines += ["", f"> 📚 **References:** {', '.join(refs)}"]

    return _md("\n".join(lines))


# ---------------------------------------------------------------------------
# B. Whiteboard Challenge cell (markdown + paired code)
# ---------------------------------------------------------------------------

def whiteboard_challenge_cell(
    *,
    title: str,
    prompt: str,
    constraints: Sequence[str],
    solution_code: str,
    complexity: str,
) -> tuple[dict, dict]:
    """Return a ``(markdown_cell, code_cell)`` tuple for a whiteboard challenge.

    The markdown cell begins with ``### 🧑‍💻 Whiteboard Challenge: {title}``
    (Requirement 20.2). The code cell contains the provided ``solution_code``
    verbatim and — if it does not already do so — is wrapped with guidance
    reminding the reader that a correct solution must call ``mx.eval`` and
    ``assert`` on its output (Requirement 4.2–4.4).
    """
    md_lines = [
        f"### 🧑‍💻 Whiteboard Challenge: {title}",
        "",
        f"**Prompt:** {prompt}",
        "",
        "**Constraints:**",
        _bullets(constraints),
        "",
        f"**Expected complexity:** {complexity}",
    ]
    md_cell = _md("\n".join(md_lines))

    # Sanity-check that the solution code obeys the verifiability rule
    # (Requirement 4.2, 4.4). We fail LOUDLY at template-build time to catch
    # authoring mistakes before the notebook ever executes.
    if "assert" not in solution_code:
        raise ValueError(
            "whiteboard_challenge_cell: solution_code must contain at least "
            "one `assert` statement (Requirement 4.2)."
        )
    if "mx.eval" not in solution_code:
        raise ValueError(
            "whiteboard_challenge_cell: solution_code must call `mx.eval` "
            "before reading lazy MLX arrays (Requirement 4.4)."
        )

    code_cell = _code(solution_code)
    return md_cell, code_cell


# ---------------------------------------------------------------------------
# C. Complexity Analysis cell (markdown)
# ---------------------------------------------------------------------------

def complexity_analysis_cell(
    *,
    op: str,
    flops: str,
    memory: str,
    latency_mlx: str,
    scaling: str,
) -> dict:
    """Markdown cell for a complexity / systems analysis.

    The cell begins with ``### 📐 Complexity & Systems: {op}``
    (Requirement 20.3) and declares FLOPs, memory, measured latency, and a
    one-sentence scaling implication (Requirement 5.4).
    """
    lines = [
        f"### 📐 Complexity & Systems: {op}",
        "",
        "| Quantity | Formula / Value | Notes |",
        "|---|---|---|",
        f"| FLOPs | `{flops}` | per forward pass |",
        f"| Memory | `{memory}` | working set |",
        f"| Latency (M4 Pro, MLX) | `{latency_mlx}` | measured, see paired benchmark cell |",
        "",
        f"💡 **Scaling implication:** {scaling}",
    ]
    return _md("\n".join(lines))


# ---------------------------------------------------------------------------
# D. Production Context cell (markdown)
# ---------------------------------------------------------------------------

def production_context_cell(
    *,
    concept: str,
    vllm: str,
    sglang: str,
    trt_llm: str,
    mlx_lm: str,
) -> dict:
    """Markdown cell tying a concept to production inference stacks.

    Begins with the ``🏭`` emoji (Requirement 20.4) and references all four
    of vLLM, SGLang, TensorRT-LLM, and MLX-LM (Requirement 2.5).
    """
    lines = [
        f"### 🏭 How Production Systems Handle This — {concept}",
        "",
        "| System | Mechanism | Notes |",
        "|---|---|---|",
        f"| vLLM | {vllm} | |",
        f"| SGLang | {sglang} | |",
        f"| TensorRT-LLM | {trt_llm} | |",
        f"| MLX-LM | {mlx_lm} | |",
        "",
        "🎯 **Interview tip:** Know at least one concrete trade-off per row.",
    ]
    return _md("\n".join(lines))


# ---------------------------------------------------------------------------
# E. Frontier Context cell (markdown)
# ---------------------------------------------------------------------------

def frontier_context_cell(
    *,
    topic: str,
    papers: Sequence[tuple[str, int, str]],
    current_sota: str,
) -> dict:
    """Markdown cell surveying the frontier work on a topic.

    Begins with ``### 🔭 Frontier Context ({topic})`` (Requirement 20.5).
    ``papers`` is a sequence of ``(title, year, one_liner)`` tuples rendered
    as a numbered paper trail.
    """
    paper_lines = [f"{title} ({year}) — {oneliner}" for title, year, oneliner in papers]
    lines = [
        f"### 🔭 Frontier Context ({topic})",
        "",
        "**Paper trail:**",
        _numbered(paper_lines),
        "",
        f"**Current SOTA:** {current_sota}",
    ]
    return _md("\n".join(lines))


# ---------------------------------------------------------------------------
# F. Debugging & Failure Modes cell (markdown + paired diagnostic code)
# ---------------------------------------------------------------------------

def debugging_failures_cell(
    *,
    symptom: str,
    root_causes: Sequence[str],
    diagnostic_code: str,
) -> tuple[dict, dict]:
    """Return a ``(markdown_cell, code_cell)`` tuple for a debugging stratum.

    Markdown begins with ``### 🛠️ Failure Modes & Debugging: {symptom}``
    (Requirement 20.6). The code cell contains the diagnostic code that
    reproduces the symptom and demonstrates the fix.
    """
    md_lines = [
        f"### 🛠️ Failure Modes & Debugging: {symptom}",
        "",
        "**Root causes (ranked by frequency):**",
        _numbered(list(root_causes)),
        "",
        "**Diagnostic code below reproduces the symptom then shows the fix:**",
    ]
    md_cell = _md("\n".join(md_lines))
    code_cell = _code(diagnostic_code)
    return md_cell, code_cell


# ---------------------------------------------------------------------------
# G. End-of-notebook Interview Question Index (markdown)
# ---------------------------------------------------------------------------

def interview_index_cell(questions: Sequence[InterviewQuestion]) -> dict:
    """Markdown cell listing every interview question in the notebook.

    Title is ``### 📋 Interview Question Index`` (Requirement 20.7). Each row
    shows id · difficulty · roles · a truncated preview of the question text.
    """
    header = [
        "### 📋 Interview Question Index",
        "",
        "| ID | Difficulty | Roles | Question |",
        "|---|---|---|---|",
    ]
    rows: list[str] = []
    for q in questions:
        qid = q["id"]
        difficulty = q["difficulty"]
        roles = _format_roles(q["roles"])
        # Keep the preview on one line: collapse whitespace, trim to 80 chars.
        preview = " ".join(q["question"].split())
        if len(preview) > 80:
            preview = preview[:77] + "..."
        rows.append(f"| `{qid}` | {difficulty} | {roles} | {preview} |")

    return _md("\n".join(header + rows))


# ---------------------------------------------------------------------------
# Helpers — benchmark + separator
# ---------------------------------------------------------------------------

def benchmark_cell(*, op: str, code: str) -> dict:
    """Code cell implementing the canonical MLX benchmark pattern.

    The produced cell includes ``import time`` / ``import mlx.core as mx``
    headers, ≥ 3 warmup iterations, a ``time.perf_counter`` timed loop, and
    a final ``mx.eval`` — satisfying Requirement 5.2 and 5.3 when paired
    with a Complexity Analysis cell.

    ``op`` is a human-readable operation label printed with the result.
    ``code`` is the body that defines a callable ``f`` taking no arguments
    and returning an MLX array; the harness calls ``f()`` in both warmup
    and timed loops.
    """
    body = f"""# Benchmark: {op}
import time
import mlx.core as mx

{code}

# Warmup — first few runs exclude compile / allocate overhead
for _ in range(3):
    _y = f()
    mx.eval(_y)

N = 10
t0 = time.perf_counter()
for _ in range(N):
    _y = f()
    mx.eval(_y)
dt_ms = (time.perf_counter() - t0) / N * 1000.0
print(f"{op}: {{dt_ms:.3f}} ms / call  (N={{N}}, 3 warmups)")
"""
    return _code(body)


def separator_cell() -> dict:
    """Return a minimal ``---`` markdown separator cell (Requirement 6.5)."""
    return _md("---")
