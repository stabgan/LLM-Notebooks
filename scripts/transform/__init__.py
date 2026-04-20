"""Transformation utilities for the interview-grade notebooks curriculum.

This package contains the cell template library and (forthcoming) transform
scripts that convert the 20 MLX-on-Apple-Silicon notebooks into a coherent
interview-prep curriculum.

All cell templates return MCP-Jupyter-compatible dicts of the form::

    {"cell_type": "markdown" | "code", "source": "<string>"}

so they can be passed directly to `mcp_jupyter_editor_ipynb_insert_cell`.
"""

from scripts.transform.templates import (
    Difficulty,
    InterviewQuestion,
    Role,
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

__all__ = [
    "Difficulty",
    "InterviewQuestion",
    "Role",
    "benchmark_cell",
    "complexity_analysis_cell",
    "debugging_failures_cell",
    "frontier_context_cell",
    "interview_index_cell",
    "interview_question_cell",
    "production_context_cell",
    "separator_cell",
    "whiteboard_challenge_cell",
]
