"""Unit tests for `scripts.transform.templates`.

One test per template function verifies:
  (a) the correct emoji prefix appears in the cell source (Requirement 20),
  (b) the `cell_type` field is correct ("markdown" or "code"),
  (c) required substrings / fields from the inputs appear in the output.

Run with:
    .venv/bin/python -m pytest tests/test_transform_templates.py -v --no-header
"""

from __future__ import annotations

import pytest

from scripts.transform.templates import (
    InterviewQuestion,
    benchmark_cell,
    complexity_analysis_cell,
    debugging_failures_cell,
    frontier_context_cell,
    interview_index_cell,
    interview_question_cell,
    production_context_cell,
    separator_cell,
    whiteboard_challenge_cell,
)


# ---------------------------------------------------------------------------
# interview_question_cell — 🎯
# ---------------------------------------------------------------------------

def test_interview_question_cell_structure_and_content():
    q: InterviewQuestion = {
        "id": "nb05-q03",
        "notebook": "05_self_attention.ipynb",
        "section": "Scaled dot-product attention",
        "difficulty": "core",
        "roles": ["mle", "research_engineer"],
        "topic_tags": ["attention", "softmax"],
        "question": "Why is attention scaled by 1/sqrt(d_k)?",
        "answer_key_points": [
            "variance of dot product grows with d_k",
            "unscaled softmax saturates, gradients vanish",
            "1/sqrt(d_k) keeps variance ≈ 1",
        ],
        "worked_solution_cell_id": None,
        "trap": "forgetting that the scaling is per-head, not per-layer",
        "references": ["https://arxiv.org/abs/1706.03762"],
        "added_in": "",
    }

    cell = interview_question_cell(q)

    # (b) cell_type
    assert cell["cell_type"] == "markdown"

    src = cell["source"]
    # (a) emoji prefix
    assert "🎯" in src
    assert src.startswith("### 🎯 Interview Question nb05-q03")

    # (c) required substrings from inputs
    assert "[core]" in src
    assert "mle, research_engineer" in src
    assert "Why is attention scaled by 1/sqrt(d_k)?" in src
    assert "variance of dot product grows with d_k" in src
    assert "unscaled softmax saturates, gradients vanish" in src
    assert "Trap:" in src
    assert "forgetting that the scaling is per-head, not per-layer" in src
    assert "https://arxiv.org/abs/1706.03762" in src


# ---------------------------------------------------------------------------
# whiteboard_challenge_cell — 🧑‍💻
# ---------------------------------------------------------------------------

def test_whiteboard_challenge_cell_returns_markdown_and_code_pair():
    solution_code = (
        "import mlx.core as mx\n"
        "def solution(x):\n"
        "    return x * 2\n"
        "x = mx.array([1.0, 2.0, 3.0])\n"
        "y = solution(x)\n"
        "mx.eval(y)\n"
        "assert y.shape == (3,), f'got {y.shape}'\n"
    )

    md_cell, code_cell = whiteboard_challenge_cell(
        title="Doubling in MLX",
        prompt="Write a function that doubles every element of an MLX array.",
        constraints=["Use MLX only", "Single-pass"],
        solution_code=solution_code,
        complexity="O(n) time, O(n) memory",
    )

    # (b) cell_type
    assert md_cell["cell_type"] == "markdown"
    assert code_cell["cell_type"] == "code"

    md_src = md_cell["source"]
    code_src = code_cell["source"]

    # (a) emoji prefix on markdown cell
    assert "🧑\u200d💻" in md_src  # zero-width joiner form
    assert md_src.startswith("### 🧑\u200d💻 Whiteboard Challenge: Doubling in MLX")

    # (c) required substrings from inputs
    assert "Write a function that doubles every element of an MLX array." in md_src
    assert "Use MLX only" in md_src
    assert "Single-pass" in md_src
    assert "O(n) time, O(n) memory" in md_src

    # Solution code passed through verbatim and satisfies verifiability rules
    assert code_src == solution_code
    assert "assert" in code_src
    assert "mx.eval" in code_src


def test_whiteboard_challenge_cell_rejects_missing_assert():
    bad_code = "import mlx.core as mx\ny = mx.array([1.0])\nmx.eval(y)\n"
    with pytest.raises(ValueError, match="assert"):
        whiteboard_challenge_cell(
            title="no assert",
            prompt="x",
            constraints=["c"],
            solution_code=bad_code,
            complexity="O(1)",
        )


def test_whiteboard_challenge_cell_rejects_missing_mx_eval():
    bad_code = "import mlx.core as mx\nassert True\n"
    with pytest.raises(ValueError, match="mx.eval"):
        whiteboard_challenge_cell(
            title="no eval",
            prompt="x",
            constraints=["c"],
            solution_code=bad_code,
            complexity="O(1)",
        )


# ---------------------------------------------------------------------------
# complexity_analysis_cell — 📐
# ---------------------------------------------------------------------------

def test_complexity_analysis_cell_structure_and_content():
    cell = complexity_analysis_cell(
        op="scaled dot-product attention",
        flops="O(B·H·T^2·D)",
        memory="O(B·H·T^2) for attn + O(B·T·H·D) for QKV",
        latency_mlx="3.2 ms @ seq_len=512",
        scaling="quadratic in T — pathological beyond 8k",
    )

    assert cell["cell_type"] == "markdown"
    src = cell["source"]

    assert "📐" in src
    assert src.startswith("### 📐 Complexity & Systems: scaled dot-product attention")

    # required substrings
    assert "O(B·H·T^2·D)" in src
    assert "O(B·H·T^2) for attn + O(B·T·H·D) for QKV" in src
    assert "3.2 ms @ seq_len=512" in src
    assert "quadratic in T — pathological beyond 8k" in src


# ---------------------------------------------------------------------------
# production_context_cell — 🏭
# ---------------------------------------------------------------------------

def test_production_context_cell_structure_and_content():
    cell = production_context_cell(
        concept="KV-cache paging",
        vllm="PagedAttention blocks",
        sglang="RadixAttention prefix sharing",
        trt_llm="KV-cache reuse + FMHA",
        mlx_lm="contiguous cache growing in-place",
    )

    assert cell["cell_type"] == "markdown"
    src = cell["source"]

    assert "🏭" in src
    assert src.startswith("### 🏭")
    assert "KV-cache paging" in src

    # (c) all four production systems and their mechanisms
    for substr in [
        "vLLM", "PagedAttention blocks",
        "SGLang", "RadixAttention prefix sharing",
        "TensorRT-LLM", "KV-cache reuse + FMHA",
        "MLX-LM", "contiguous cache growing in-place",
    ]:
        assert substr in src, f"missing: {substr!r}"


# ---------------------------------------------------------------------------
# frontier_context_cell — 🔭
# ---------------------------------------------------------------------------

def test_frontier_context_cell_structure_and_content():
    cell = frontier_context_cell(
        topic="Test-time compute",
        papers=[
            ("OpenAI o1", 2024, "RL-trained CoT with hidden scratchpad"),
            ("DeepSeek-R1", 2025, "pure-RL reasoning, GRPO at scale"),
        ],
        current_sota="o3 + verifier-guided search; R1-Zero on math/code",
    )

    assert cell["cell_type"] == "markdown"
    src = cell["source"]

    assert "🔭" in src
    assert src.startswith("### 🔭 Frontier Context (Test-time compute)")

    # (c) both papers and the SOTA line
    assert "OpenAI o1" in src
    assert "2024" in src
    assert "RL-trained CoT with hidden scratchpad" in src
    assert "DeepSeek-R1" in src
    assert "2025" in src
    assert "pure-RL reasoning, GRPO at scale" in src
    assert "o3 + verifier-guided search; R1-Zero on math/code" in src


# ---------------------------------------------------------------------------
# debugging_failures_cell — 🛠️
# ---------------------------------------------------------------------------

def test_debugging_failures_cell_returns_markdown_and_code_pair():
    diagnostic = (
        "import mlx.core as mx\n"
        "x = mx.array([1.0])\n"
        "# forgot mx.eval(x) — lazy array stays unevaluated\n"
        "mx.eval(x)\n"
        "print(x)\n"
    )
    md_cell, code_cell = debugging_failures_cell(
        symptom="NaN loss after 100 steps",
        root_causes=[
            "missing loss scaling in bfloat16",
            "learning-rate warmup too short",
        ],
        diagnostic_code=diagnostic,
    )

    assert md_cell["cell_type"] == "markdown"
    assert code_cell["cell_type"] == "code"

    md_src = md_cell["source"]
    # (a) emoji prefix
    assert "🛠️" in md_src
    assert md_src.startswith("### 🛠️ Failure Modes & Debugging: NaN loss after 100 steps")

    # (c) required substrings
    assert "missing loss scaling in bfloat16" in md_src
    assert "learning-rate warmup too short" in md_src
    assert code_cell["source"] == diagnostic


# ---------------------------------------------------------------------------
# interview_index_cell — 📋
# ---------------------------------------------------------------------------

def test_interview_index_cell_structure_and_content():
    questions: list[InterviewQuestion] = [
        {
            "id": "nb00-q01",
            "notebook": "00_environment_apple_silicon.ipynb",
            "section": "Unified memory",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["memory"],
            "question": "What does 'unified memory' mean on Apple Silicon?",
            "answer_key_points": ["CPU and GPU share pointers", "no copy cost",
                                  "bandwidth-limited by DRAM"],
            "worked_solution_cell_id": None,
            "trap": None,
            "references": [],
            "added_in": "",
        },
        {
            "id": "nb00-q02",
            "notebook": "00_environment_apple_silicon.ipynb",
            "section": "Lazy eval",
            "difficulty": "core",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["mlx"],
            "question": "When must you call mx.eval explicitly?",
            "answer_key_points": ["before host reads", "before timing",
                                  "to break cyclic graphs"],
            "worked_solution_cell_id": None,
            "trap": None,
            "references": [],
            "added_in": "",
        },
    ]

    cell = interview_index_cell(questions)

    assert cell["cell_type"] == "markdown"
    src = cell["source"]

    # (a) title with emoji
    assert "📋" in src
    assert src.startswith("### 📋 Interview Question Index")

    # (c) both rows present with their ids, difficulties, roles, and previews
    assert "`nb00-q01`" in src
    assert "warmup" in src
    assert "mle" in src
    assert "unified memory" in src.lower()

    assert "`nb00-q02`" in src
    assert "core" in src
    assert "research_engineer, systems_engineer" in src
    assert "When must you call mx.eval explicitly?" in src


# ---------------------------------------------------------------------------
# benchmark_cell — helper
# ---------------------------------------------------------------------------

def test_benchmark_cell_structure_and_content():
    body = (
        "import mlx.core as mx\n"
        "x = mx.random.normal(shape=(128, 128))\n"
        "def f():\n"
        "    return x @ x\n"
    )
    cell = benchmark_cell(op="matmul 128x128", code=body)

    assert cell["cell_type"] == "code"
    src = cell["source"]

    # (c) required patterns: perf_counter, mx.eval, ≥3 warmups, op label
    assert "time.perf_counter" in src
    assert "mx.eval" in src
    assert "for _ in range(3):" in src  # warmups
    assert "matmul 128x128" in src
    # The provided body is embedded verbatim
    assert "x = mx.random.normal(shape=(128, 128))" in src
    assert "return x @ x" in src


# ---------------------------------------------------------------------------
# separator_cell — helper
# ---------------------------------------------------------------------------

def test_separator_cell_is_dashes_markdown():
    cell = separator_cell()
    assert cell["cell_type"] == "markdown"
    assert cell["source"] == "---"
