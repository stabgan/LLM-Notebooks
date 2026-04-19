# Implementation Plan: Interview-Grade Notebooks

## Overview

Transform 20 existing MLX notebooks into an interview-prep curriculum via additive, idempotent Python transforms. Foundation first (templates, Q-bank, verifier, pipeline runner, property tests), then per-notebook transforms batched in parallel tiers of ≤10, then curriculum-wide finalization.

**Implementation Language:** Python 3 (MLX ≥ 0.18 on Apple Silicon). No PyTorch / TensorFlow / JAX anywhere in new code.

## Execution Strategy

**Batching model — 4 tiers × 5 notebooks, run as 2 parallel waves of 10:**

```
Wave A (parallel, max_concurrency=10):  Foundations (00-04) + Architecture (05-09)
Wave B (parallel, max_concurrency=10):  Systems      (10-14) + Frontier     (15-19)
```

- Tasks 1–6 are **prerequisites**; must complete sequentially before any notebook transform.
- Tasks 7–16 (Wave A) run in parallel after task 6.
- Tasks 17–26 (Wave B) run in parallel after Wave A fully commits.
- Tasks 27–30 run sequentially after all notebooks are transformed.
- Each per-notebook task is self-contained: transform → execute → pytest → Q-bank → review → commit → push.
- `question_bank.json` writes are coordinated via file lock (Requirement 12.2); each transform owns only its `nb{NN}-q*` slice (Requirement 12.3).
- Re-runs are safe (idempotent by emoji/id detection — Requirement 10).

All commands invoke `.venv/bin/python scripts/transform/...`; no `python -c` inline code (Requirement 11.3).

---

## Tasks

### Foundation (sequential, before any notebook work)

- [ ] 1. Build the cell template library
  - Create `scripts/transform/__init__.py` and `scripts/transform/templates.py`
  - Implement `interview_question_cell`, `whiteboard_challenge_cell`, `complexity_analysis_cell`, `production_context_cell`, `frontier_context_cell`, `debugging_failures_cell` as pure functions returning MCP-compatible cell dicts
  - Enforce emoji prefixes: 🎯 🧑‍💻 📐 🏭 🔭 🛠️
  - Add `interview_index_cell` helper for the `📋 Interview Question Index` end-of-notebook cell
  - Include `TypedDict` schemas for `InterviewQuestion`, `Difficulty`, `Role`
  - _Validates: Requirements 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7_
  - [ ]* 1.1 Write unit tests for each template function
    - One test per template producing exact expected markdown/code dict
    - _Validates: Requirement 20.1–20.7_

- [ ] 2. Build the question-bank infrastructure
  - Create empty `.kiro/specs/interview-grade-notebooks/question_bank.json` as `{"questions": []}`
  - Create `scripts/transform/qbank.py` with: `load()`, `save()`, `upsert_slice(nb_num, records)`, `delete_slice(nb_num)`, `filter_by_role(role)`, `validate_schema(record)`
  - Implement `filelock`-backed write coordination (one `.lock` file next to `question_bank.json`)
  - Enforce id pattern `nb{NN}-q{NN}`, difficulty ∈ {warmup, core, stretch, research}, roles ⊆ {mle, research_engineer, systems_engineer}, 3 ≤ len(answer_key_points) ≤ 7
  - _Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9, 12.2, 12.3, 12.4, 17.1, 17.2_

- [ ] 3. Build the verification harness
  - Create `scripts/transform/verify.py` with `verify_notebook(nb_num)`:
    - Run `jupyter nbconvert --to notebook --execute --inplace` on target notebook; capture failing cell index
    - Run `.venv/bin/python -m pytest tests/ -v --no-header` (non-interactive); assert ≥ 34 passing
    - Run bijection check: every 🎯 cell id ↔ exactly one Q-bank record
    - Exit non-zero with actionable message on any failure (print failing cell idx, list drift ids, print peak MLX memory on OOM)
  - _Validates: Requirements 7.1, 7.2, 8.1, 8.2, 8.3, 8.4, 21.1, 21.2, 21.3_

- [ ] 4. Build the pipeline runner
  - Create `scripts/transform/run_pipeline.py` (argparse: `--notebook NN`, `--all`, `--parallel N` (default 10), `--push`, `--skip-review`)
  - Per-notebook flow: `nb{NN}.py` → `verify.py` → `commit.py` (stage + conventional commit `feat(nb{NN}): {stage}` + `git push origin main`)
  - Parallelism: `concurrent.futures.ProcessPoolExecutor(max_workers=min(parallel, 10))`
  - Create `scripts/transform/commit.py` with `commit_and_push(nb_num, stage)`; captures returned SHA for Q-bank `added_in` backfill
  - Reject any invocation that would pass `-c` (assert not in `sys.argv`)
  - _Validates: Requirements 11.1, 11.2, 11.3, 11.4, 12.1, 16.1, 16.2, 16.3, 16.4, 21.4_

- [ ] 5. Add new property-based tests for interview-grade invariants
  - Create `tests/test_interview_grade_properties.py` with hypothesis strategies that sample notebooks and Q-bank records
  - `@settings(max_examples=100)` on every test
  - [ ] 5.1 `test_coverage_floors_and_spread` (Property 1)
    - _Validates: Requirements 1.1–1.8_
  - [ ] 5.2 `test_question_bank_schema` (Property 3)
    - _Validates: Requirements 3.2–3.6_
  - [ ] 5.3 `test_question_bank_bijection` (Property 4)
    - _Validates: Requirements 3.7, 3.8_
  - [ ] 5.4 `test_role_filter_correctness` (Property 5)
    - _Validates: Requirements 3.9, 17.2_
  - [ ] 5.5 `test_whiteboard_verifiability` (Property 6)
    - _Validates: Requirements 4.1, 4.2, 4.3, 4.4, 7.5_
  - [ ] 5.6 `test_complexity_cell_faithfulness` (Property 7)
    - _Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ] 5.7 `test_narrative_preservation` (Property 8)
    - _Validates: Requirements 6.1, 6.2, 6.3, 6.5_
  - [ ] 5.8 `test_mlx_only_new_code` (Property 9)
    - _Validates: Requirement 7.4_
  - [ ] 5.9 `test_transform_idempotence` (Property 10)
    - _Validates: Requirements 10.1, 10.2, 10.3, 10.4, 21.4_
  - [ ] 5.10 `test_cell_template_consistency` (Property 12)
    - _Validates: Requirements 20.1–20.7_
  - [ ] 5.11 `test_hypothesis_iteration_floor` (Property 13) — meta-test
    - _Validates: Requirement 9.5_
  - [ ] 5.12 `test_no_inline_python_in_pipeline` (Property 16)
    - _Validates: Requirement 11.3_

- [ ] 6. Build the notebook manifest loader
  - Create `scripts/transform/manifests.py` with `NotebookManifest` pydantic model (fields per design D3)
  - Create `scripts/transform/manifests/` directory with 20 JSON manifests (`nb00.json` … `nb19.json`) populated from LLD-4 table: prev score, target counts per stratum, `preserve_sections`, `anchor_after_headings`
  - Implement `load_manifest(nb_num) -> NotebookManifest`
  - _Validates: Requirements 6.2, 6.4, 10.3_

### Checkpoint A

- [ ] 6.5 Checkpoint — foundation tests pass
  - Run `.venv/bin/python -m pytest tests/ -v --no-header` locally; confirm existing 34+ tests still pass and new tests that do not depend on transformed notebooks pass or xfail cleanly
  - Ensure all tests pass, ask the user if questions arise.

---

### Wave A — Foundations + Architecture (parallel, max 10 concurrent)

_Each task below creates `scripts/transform/nb{NN}.py`, runs the transform, executes the notebook, runs pytest, writes the Q-bank slice, invokes the pedagogy-reviewer, then commits and pushes._

- [ ] 7. Transform NB 00 (Environment)
  - [ ] 7.1 Create `scripts/transform/nb00.py` with anchor-by-heading insertions
    - 6 🎯 interview Qs (Apple Silicon memory model, `mx.metal` API, MLX lazy-eval)
    - 2 🧑‍💻 whiteboard challenges (with `mx.eval` + `assert` solution cells)
    - 2 📐 complexity cells (with paired benchmark code cells — `time.perf_counter` + ≥3 warmups)
    - 1 🏭 production context, 1 🔭 frontier context, 1 🛠️ debugging cell
    - 1 `📋 Interview Question Index` end cell
    - Idempotency guard: skip insertion if 🎯 emoji prefix already present at target index
    - _Validates: Requirements 1.1–1.8, 2.1, 4.1–4.5, 5.1–5.4, 6.1, 6.4, 6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 7.2 Run transform via `run_pipeline.py --notebook 00`
    - Invokes `nb00.py` then `verify.py` then `commit.py`
    - _Validates: Requirements 11.1, 11.2_
  - [ ] 7.3 Execute notebook end-to-end on MLX
    - `jupyter nbconvert --to notebook --execute --inplace 00_environment_apple_silicon.ipynb`
    - _Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ] 7.4 Run pytest after transform
    - `.venv/bin/python -m pytest tests/ -v` — must report ≥ 34 passing
    - _Validates: Requirements 8.1, 8.2, 8.3, 8.4_
  - [ ] 7.5 Add Q-bank entries for nb00
    - Upsert `nb00-q01` … `nb00-qNN` into `question_bank.json` via `qbank.upsert_slice(0, ...)`
    - Backfill `added_in` with the commit SHA produced in 7.7
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 7.6 Invoke pedagogy-reviewer sub-agent on nb00
    - Persist review at `.kiro/specs/interview-grade-notebooks/reviews/nb00.md`
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 7.7 Commit & push nb00
    - Conventional commit: `feat(nb00): interview-grade transform`; push to `origin main`
    - _Validates: Requirements 16.1, 16.2, 16.3_

- [ ] 8. Transform NB 01 (MLX fundamentals)
  - [ ] 8.1 Create `scripts/transform/nb01.py` (lazy-eval traps, compile graph, shape-broadcasting whiteboards)
    - _Validates: Requirements 1.1–1.8, 2.1, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 8.2 Run transform via `run_pipeline.py --notebook 01`
  - [ ] 8.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 8.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 8.5 Add Q-bank entries for nb01
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 8.6 Invoke pedagogy-reviewer sub-agent on nb01
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 8.7 Commit & push nb01
    - _Validates: Requirements 16.1–16.3_

- [ ] 9. Transform NB 02 (Math foundations)
  - [ ] 9.1 Create `scripts/transform/nb02.py` (softmax Jacobian, cross-entropy gradient, KL properties, matrix calc)
    - _Validates: Requirements 1.1–1.8, 2.1, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 9.2 Run transform via `run_pipeline.py --notebook 02`
  - [ ] 9.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 9.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 9.5 Add Q-bank entries for nb02
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 9.6 Invoke pedagogy-reviewer sub-agent on nb02
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 9.7 Commit & push nb02
    - _Validates: Requirements 16.1–16.3_

- [ ] 10. Transform NB 03 (Tokenization)
  - [ ] 10.1 Create `scripts/transform/nb03.py` (BPE vs SentencePiece vs tiktoken, leakage traps, vocab-size math)
    - _Validates: Requirements 1.1–1.8, 2.1, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 10.2 Run transform via `run_pipeline.py --notebook 03`
  - [ ] 10.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 10.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 10.5 Add Q-bank entries for nb03
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 10.6 Invoke pedagogy-reviewer sub-agent on nb03
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 10.7 Commit & push nb03
    - _Validates: Requirements 16.1–16.3_

- [ ] 11. Transform NB 04 (Embeddings & Positional Encodings)
  - [ ] 11.1 Create `scripts/transform/nb04.py` (RoPE derivation, ALiBi slopes, YaRN/NTK-aware)
    - _Validates: Requirements 1.1–1.8, 2.1, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 11.2 Run transform via `run_pipeline.py --notebook 04`
  - [ ] 11.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 11.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 11.5 Add Q-bank entries for nb04
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 11.6 Invoke pedagogy-reviewer sub-agent on nb04
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 11.7 Commit & push nb04
    - _Validates: Requirements 16.1–16.3_

- [ ] 12. Transform NB 05 (Self-attention)
  - [ ] 12.1 Create `scripts/transform/nb05.py` (softmax-scaling derivation, causal mask whiteboard, attn memory formula)
    - _Validates: Requirements 1.1–1.8, 2.2, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 12.2 Run transform via `run_pipeline.py --notebook 05`
  - [ ] 12.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 12.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 12.5 Add Q-bank entries for nb05
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 12.6 Invoke pedagogy-reviewer sub-agent on nb05
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 12.7 Commit & push nb05
    - _Validates: Requirements 16.1–16.3_

- [ ] 13. Transform NB 06 (Transformer)
  - [ ] 13.1 Create `scripts/transform/nb06.py` (pre-norm vs post-norm, parameter-count formulas, residual scaling)
    - _Validates: Requirements 1.1–1.8, 2.2, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 13.2 Run transform via `run_pipeline.py --notebook 06`
  - [ ] 13.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 13.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 13.5 Add Q-bank entries for nb06
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 13.6 Invoke pedagogy-reviewer sub-agent on nb06
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 13.7 Commit & push nb06
    - _Validates: Requirements 16.1–16.3_

- [ ] 14. Transform NB 07 (Building GPT)
  - [ ] 14.1 Create `scripts/transform/nb07.py` (KV-cache memory growth, generation strategies greedy/top-k/top-p/beam)
    - _Validates: Requirements 1.1–1.8, 2.2, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 14.2 Run transform via `run_pipeline.py --notebook 07`
  - [ ] 14.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 14.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 14.5 Add Q-bank entries for nb07
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 14.6 Invoke pedagogy-reviewer sub-agent on nb07
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 14.7 Commit & push nb07
    - _Validates: Requirements 16.1–16.3_

- [ ] 15. Transform NB 08 (Training)
  - [ ] 15.1 Create `scripts/transform/nb08.py` (LR schedules, grad clipping, mixed-precision / loss scaling)
    - _Validates: Requirements 1.1–1.8, 2.2, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 15.2 Run transform via `run_pipeline.py --notebook 08`
  - [ ] 15.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 15.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 15.5 Add Q-bank entries for nb08
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 15.6 Invoke pedagogy-reviewer sub-agent on nb08
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 15.7 Commit & push nb08
    - _Validates: Requirements 16.1–16.3_

- [ ] 16. Transform NB 09 (Modern architectures)
  - [ ] 16.1 Create `scripts/transform/nb09.py` (LLaMA vs Gemma vs Mistral, GQA/MQA memory, SwiGLU vs GeGLU)
    - _Validates: Requirements 1.1–1.8, 2.2, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 16.2 Run transform via `run_pipeline.py --notebook 09`
  - [ ] 16.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 16.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 16.5 Add Q-bank entries for nb09
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 16.6 Invoke pedagogy-reviewer sub-agent on nb09
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 16.7 Commit & push nb09
    - _Validates: Requirements 16.1–16.3_

### Checkpoint B — Wave A complete

- [ ] 16.5 Checkpoint — Wave A verification
  - Run `.venv/bin/python scripts/transform/verify.py --notebooks 00-09` (batch bijection + pytest)
  - Confirm Q-bank has ≥ 60 entries (≥6 per notebook × 10 notebooks)
  - Ensure all tests pass, ask the user if questions arise.

---

### Wave B — Systems + Frontier (parallel, max 10 concurrent)

- [ ] 17. Transform NB 10 (Metal kernels)
  - [ ] 17.1 Create `scripts/transform/nb10.py` (kernel launch overhead, shared-memory tiling, roofline plots for M4 Pro; ≥2 seq-len/batch points on each 📐 cell)
    - _Validates: Requirements 1.1–1.8, 2.3, 4.1–4.5, 5.1–5.5, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 17.2 Run transform via `run_pipeline.py --notebook 10`
  - [ ] 17.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 17.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 17.5 Add Q-bank entries for nb10
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 17.6 Invoke pedagogy-reviewer sub-agent on nb10
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 17.7 Commit & push nb10
    - _Validates: Requirements 16.1–16.3_

- [ ] 18. Transform NB 11 (Inference optimization)
  - [ ] 18.1 Create `scripts/transform/nb11.py` (prefill vs decode FLOPs, speculative decoding, quant error bounds; ≥2 seq-len/batch points)
    - _Validates: Requirements 1.1–1.8, 2.3, 2.5, 4.1–4.5, 5.1–5.5, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 18.2 Run transform via `run_pipeline.py --notebook 11`
  - [ ] 18.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 18.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 18.5 Add Q-bank entries for nb11
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 18.6 Invoke pedagogy-reviewer sub-agent on nb11
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 18.7 Commit & push nb11
    - _Validates: Requirements 16.1–16.3_

- [ ] 19. Transform NB 12 (Flash / Paged / Ring attention)
  - [ ] 19.1 Create `scripts/transform/nb12.py` (FA-1/FA-2 tile math, block-table paging, ring-attention partitioning; ≥2 seq-len/batch points)
    - _Validates: Requirements 1.1–1.8, 2.3, 4.1–4.5, 5.1–5.5, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 19.2 Run transform via `run_pipeline.py --notebook 12`
  - [ ] 19.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 19.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 19.5 Add Q-bank entries for nb12
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 19.6 Invoke pedagogy-reviewer sub-agent on nb12
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 19.7 Commit & push nb12
    - _Validates: Requirements 16.1–16.3_

- [ ] 20. Transform NB 13 (Serving)
  - [ ] 20.1 Create `scripts/transform/nb13.py` (continuous batching, vLLM scheduler Qs, SLO math p50/p99; ≥2 batch-size points; 🏭 cell cites ≥3 of vLLM/SGLang/TRT-LLM/MLX-LM)
    - _Validates: Requirements 1.1–1.8, 2.3, 2.5, 4.1–4.5, 5.1–5.5, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 20.2 Run transform via `run_pipeline.py --notebook 13`
  - [ ] 20.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 20.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 20.5 Add Q-bank entries for nb13
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 20.6 Invoke pedagogy-reviewer sub-agent on nb13
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 20.7 Commit & push nb13
    - _Validates: Requirements 16.1–16.3_

- [ ] 21. Transform NB 14 (Capstone Gemma 4)
  - [ ] 21.1 Create `scripts/transform/nb14.py` (full-stack debugging challenge, production readiness checklist; ≥2 seq-len/batch points)
    - _Validates: Requirements 1.1–1.8, 2.3, 4.1–4.5, 5.1–5.5, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 21.2 Run transform via `run_pipeline.py --notebook 14`
  - [ ] 21.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 21.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 21.5 Add Q-bank entries for nb14
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 21.6 Invoke pedagogy-reviewer sub-agent on nb14
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 21.7 Commit & push nb14
    - _Validates: Requirements 16.1–16.3_

- [ ] 22. Transform NB 15 (MoE)
  - [ ] 22.1 Create `scripts/transform/nb15.py` (top-k routing math, load-balancing loss, expert-parallel memory; 🔭 cell cites 2024–2026 works)
    - _Validates: Requirements 1.1–1.8, 2.4, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 22.2 Run transform via `run_pipeline.py --notebook 15`
  - [ ] 22.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 22.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 22.5 Add Q-bank entries for nb15
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 22.6 Invoke pedagogy-reviewer sub-agent on nb15
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 22.7 Commit & push nb15
    - _Validates: Requirements 16.1–16.3_

- [ ] 23. Transform NB 16 (SSM / Mamba)
  - [ ] 23.1 Create `scripts/transform/nb16.py` (Mamba recurrence derivation, selective-scan FLOPs, Transformer-vs-SSM benchmark; 🔭 cites Mamba-2)
    - _Validates: Requirements 1.1–1.8, 2.4, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 23.2 Run transform via `run_pipeline.py --notebook 16`
  - [ ] 23.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 23.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 23.5 Add Q-bank entries for nb16
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 23.6 Invoke pedagogy-reviewer sub-agent on nb16
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 23.7 Commit & push nb16
    - _Validates: Requirements 16.1–16.3_

- [ ] 24. Transform NB 17 (Alignment: DPO/KTO/GRPO)
  - [ ] 24.1 Create `scripts/transform/nb17.py` (DPO → KTO → GRPO derivations, reward hacking debugging, β sensitivity; 🔭 cites DeepSeek-R1/GRPO)
    - _Validates: Requirements 1.1–1.8, 2.4, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 24.2 Run transform via `run_pipeline.py --notebook 17`
  - [ ] 24.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 24.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 24.5 Add Q-bank entries for nb17
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 24.6 Invoke pedagogy-reviewer sub-agent on nb17
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 24.7 Commit & push nb17
    - _Validates: Requirements 16.1–16.3_

- [ ] 25. Transform NB 18 (Scaling laws)
  - [ ] 25.1 Create `scripts/transform/nb18.py` (Chinchilla optimization, Kaplan debate, inference-optimal vs compute-optimal; 🔭 cites 2024–2026 works)
    - _Validates: Requirements 1.1–1.8, 2.4, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 25.2 Run transform via `run_pipeline.py --notebook 18`
  - [ ] 25.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 25.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 25.5 Add Q-bank entries for nb18
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 25.6 Invoke pedagogy-reviewer sub-agent on nb18
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 25.7 Commit & push nb18
    - _Validates: Requirements 16.1–16.3_

- [ ] 26. Transform NB 19 (Reasoning / Test-time compute)
  - [ ] 26.1 Create `scripts/transform/nb19.py` (CoT vs ToT, o1/o3/R1 test-time compute math, verifier-guided search; 🔭 cites o1/o3/R1)
    - _Validates: Requirements 1.1–1.8, 2.4, 4.1–4.5, 5.1–5.4, 6.1–6.5, 10.1–10.4, 20.1–20.7_
  - [ ] 26.2 Run transform via `run_pipeline.py --notebook 19`
  - [ ] 26.3 Execute notebook end-to-end on MLX
    - _Validates: Requirements 7.1–7.5_
  - [ ] 26.4 Run pytest (≥ 34 passing)
    - _Validates: Requirements 8.1–8.4_
  - [ ] 26.5 Add Q-bank entries for nb19
    - _Validates: Requirements 3.1–3.9, 12.3, 16.4_
  - [ ]* 26.6 Invoke pedagogy-reviewer sub-agent on nb19
    - _Validates: Requirements 13.1, 13.4, 14.1, 14.2_
  - [ ] 26.7 Commit & push nb19
    - _Validates: Requirements 16.1–16.3_

### Checkpoint C — Wave B complete

- [ ] 26.5 Checkpoint — Wave B verification
  - Run `.venv/bin/python scripts/transform/verify.py --notebooks 10-19`
  - Confirm Q-bank has ≥ 120 entries total across all 20 notebooks
  - Ensure all tests pass, ask the user if questions arise.

---

### Finalization (sequential)

- [ ] 27. Run parallel pedagogy review on all 20 notebooks
  - Create `scripts/transform/review_all.py` that invokes the pedagogy-reviewer sub-agent on every notebook concurrently (`max_workers=10`, respects Requirement 12.1)
  - Persist each review to `.kiro/specs/interview-grade-notebooks/reviews/nb{NN}.md`
  - Aggregate scores into `.kiro/specs/interview-grade-notebooks/reviews/summary.json` keyed by `nb{NN}` with the 6-dimension rubric breakdown
  - Fail the task if any notebook scores < 48/60 total OR any single dimension < 6/10; flag for rework
  - _Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 14.1, 14.2, 14.3, 15.1, 15.2, 15.3_

- [ ] 28. Generate the final interview question index summary
  - Create `scripts/transform/generate_index.py` that reads `question_bank.json` and emits:
    - One consolidated in-notebook `📋 Interview Question Index` cell at the end of each notebook (already present from per-notebook transforms — verified/refreshed here)
    - A role-filtered console report (`--role mle|research_engineer|systems_engineer`) as validation of the filtering logic
  - No standalone `READING_PATHS.md` is produced (Requirements 17.3, 18.2)
  - _Validates: Requirements 17.1, 17.2, 17.3, 18.2, 20.7_

- [ ] 29. Update README with curriculum structure
  - Append an `Interview-Prep Curriculum` section to `README.md` describing: the 4 tiers, the 6 cell strata + emojis, the question bank location, how to filter by role, how to run the pipeline (`run_pipeline.py`), and the quality bar (≥ 48/60, no dim < 6)
  - Include exact invocation: `.venv/bin/python scripts/transform/run_pipeline.py --all --parallel 10 --push`
  - Add a pointer to the question bank and reviews directory
  - _Validates: Requirements 11.4, 17.1, 19.5_

- [ ] 30. Final checkpoint — full-pipeline verification on all 20 notebooks
  - Run `.venv/bin/python scripts/transform/verify.py --all` (executes all 20 notebooks, runs pytest, runs bijection across the full Q-bank)
  - Confirm pytest reports ≥ 34 existing + all new interview-grade properties passing
  - Confirm Q-bank bijection holds across all 20 notebooks (no drift on either side)
  - Confirm pedagogy summary shows every notebook ≥ 48/60 with all dimensions ≥ 6/10
  - Verify no `READING_PATHS.md` file exists (Requirement 18.2)
  - Verify every notebook's `preserve_sections` cells are byte-identical to pre-transform state (`git log -p` comparison or stored hashes)
  - Ensure all tests pass, ask the user if questions arise.
  - _Validates: Requirements 6.3, 7.1, 8.2, 9.1–9.5, 13.2, 14.1, 14.2, 15.1, 15.2, 15.3, 18.2, 21.2_

---

## Notes

- Tasks marked with `*` are optional (test/review sub-tasks); core implementation, execution, Q-bank, commit & push sub-tasks are mandatory.
- Every task references its validating requirements for traceability.
- Checkpoints (6.5, 16.5, 26.5, 30) gate progression between waves.
- Per-notebook tasks 7–26 are GROUPABLE: tasks 7–16 run as Wave A (parallel, ≤10 concurrent), tasks 17–26 run as Wave B (parallel, ≤10 concurrent).
- Q-bank writes are coordinated by file lock; each transform owns only its `nb{NN}-q*` slice.
- All execution is via Python scripts under `scripts/transform/` — no inline `python -c`.
- MLX only: new code cells never import `torch`, `tensorflow`, or `jax`.
- Preserved beginner content is byte-identical pre- and post-transform (enforced by `test_narrative_preservation`).
