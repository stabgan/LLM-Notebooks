# Requirements Document

## Introduction

This feature transforms the 20 existing MLX-on-Apple-Silicon LLM notebooks (`00_environment_apple_silicon.ipynb` through `19_reasoning_test_time_compute.ipynb`) into a single coherent **interview-prep curriculum** for ML Research Engineer, ML Engineer, and ML Systems Engineer roles at frontier labs. Transformations are **additive and surgical**: all beginner-friendly content is preserved byte-identically, and six standardized cell strata (Interview Q&A, Whiteboard Challenges, Complexity Analysis, Production Context, Frontier Context, Debugging & Failure Modes) are layered on top. All new code runs on MLX on Apple Silicon, all existing property tests continue to pass, and every transformation is driven by idempotent Python scripts under `scripts/transform/`.

## Glossary

- **Notebook**: A `.ipynb` file under the repository root numbered `00` through `19`.
- **Tier**: One of four notebook groupings — Foundations (00–04), Architecture (05–09), Systems (10–14), Frontier (15–19).
- **Cell Stratum**: A category of added cell — Interview Question, Whiteboard Challenge, Complexity Analysis, Production Context, Frontier Context, or Debugging & Failures.
- **Question_Bank**: The JSON file `.kiro/specs/interview-grade-notebooks/question_bank.json` containing one record per interview question across all notebooks.
- **Transform_Script**: A Python module under `scripts/transform/` that applies a defined set of cell insertions to a single notebook.
- **Verification_Harness**: `scripts/transform/verify.py`, which executes the notebook, runs the test suite, and validates Q-bank bijection.
- **Pedagogy_Reviewer**: The sub-agent defined in `.kiro/agents/pedagogy-reviewer.md` that scores a notebook on 10 dimensions (1–10 each).
- **Quality_Rubric**: A 6-dimension × 0–10 scoring scheme (Technical Depth, Interview Breadth, Systems Rigor, Frontier Currency, Production Anchoring, Failure-Mode Literacy) totalling /60.
- **Interview_Ready_Bar**: Total score ≥ 48/60 with no dimension below 6/10.
- **Preserved_Section**: A heading listed in a notebook's manifest whose cells must remain byte-identical to the pre-transform state.
- **MLX_Runtime**: MLX ≥ 0.18 on Apple Silicon hardware (M4 Pro reference).

## Requirements

### Requirement 1: Per-Notebook Coverage Floors

**User Story:** As an interview candidate, I want every notebook to contain a consistent minimum of interview-oriented content, so that I can rely on any notebook as a complete study unit.

#### Acceptance Criteria

1. FOR EACH Notebook, THE Transform_Script SHALL insert at least 6 Interview Question cells.
2. FOR EACH Notebook, THE Transform_Script SHALL insert at least 2 Whiteboard Challenge cells.
3. FOR EACH Notebook, THE Transform_Script SHALL insert at least 2 Complexity Analysis cells.
4. FOR EACH Notebook, THE Transform_Script SHALL insert at least 1 Production Context cell.
5. FOR EACH Notebook, THE Transform_Script SHALL insert at least 1 Frontier Context cell.
6. FOR EACH Notebook, THE Transform_Script SHALL insert at least 1 Debugging & Failure Modes cell.
7. FOR EACH Notebook, THE Interview Question cells SHALL collectively cover all four difficulty tiers (`warmup`, `core`, `stretch`, `research`).
8. FOR EACH Notebook, THE Interview Question cells SHALL collectively cover all three roles (`mle`, `research_engineer`, `systems_engineer`).

### Requirement 2: Tier-Specific Content Emphasis

**User Story:** As a learner progressing through the curriculum, I want each tier to emphasize the skills relevant to its layer of the stack, so that content depth matches the topic.

#### Acceptance Criteria

1. FOR notebooks 00–04 (Foundations tier), THE Transform_Script SHALL include Interview Question cells covering math derivations, MLX lazy-eval semantics, tokenizer tradeoffs, and embedding geometry.
2. FOR notebooks 05–09 (Architecture tier), THE Transform_Script SHALL include Interview Question cells covering attention derivations, positional encoding variants, GQA/MQA memory math, and modern architecture choices.
3. FOR notebooks 10–14 (Systems tier), THE Transform_Script SHALL include Complexity Analysis cells covering Metal kernel patterns, KV-cache memory formulas, FlashAttention tiling, and PagedAttention.
4. FOR notebooks 15–19 (Frontier tier), THE Transform_Script SHALL include Frontier Context cells citing 2024–2026 works (DeepSeek-R1, Gemma 4, Claude 3.5, o1/o3, Mamba-2, GRPO).
5. FOR EACH Production Context cell, THE Transform_Script SHALL reference at least three of: vLLM, SGLang, TensorRT-LLM, MLX-LM.

### Requirement 3: Question Bank Schema and Bijection

**User Story:** As a curriculum maintainer, I want every interview question to be indexed in a single JSON file with a stable schema, so that questions are programmatically searchable, role-filterable, and verifiable.

#### Acceptance Criteria

1. THE Question_Bank SHALL exist at `.kiro/specs/interview-grade-notebooks/question_bank.json`.
2. FOR EACH record in the Question_Bank, THE record SHALL contain the fields `id`, `notebook`, `section`, `difficulty`, `roles`, `topic_tags`, `question`, `answer_key_points`, `worked_solution_cell_id`, `trap`, `references`, and `added_in`.
3. FOR EACH record in the Question_Bank, THE `id` field SHALL match the pattern `nb{NN}-q{NN}` where `{NN}` is a two-digit number.
4. FOR EACH record in the Question_Bank, THE `difficulty` field SHALL be one of `warmup`, `core`, `stretch`, `research`.
5. FOR EACH record in the Question_Bank, THE `roles` field SHALL be a non-empty subset of `{mle, research_engineer, systems_engineer}`.
6. FOR EACH record in the Question_Bank, THE `answer_key_points` field SHALL contain between 3 and 7 items inclusive.
7. FOR EACH Interview Question cell in any Notebook, THE Question_Bank SHALL contain exactly one record whose `id` matches the cell's question id (forward bijection).
8. FOR EACH record in the Question_Bank, THE corresponding Notebook SHALL contain exactly one Interview Question cell whose question id matches the record's `id` (reverse bijection).
9. THE Question_Bank SHALL support role filtering such that filtering by a role returns only records whose `roles` field includes that role.

### Requirement 4: Whiteboard Challenge Verifiability

**User Story:** As a learner solving a whiteboard challenge, I want every challenge to have a runnable, verified solution cell, so that I can confirm correctness against a known-good reference.

#### Acceptance Criteria

1. FOR EACH Whiteboard Challenge cell, THE Transform_Script SHALL insert an immediately-following code cell containing a solution function.
2. FOR EACH whiteboard solution code cell, THE code cell SHALL contain at least one `assert` statement that validates the solution's output.
3. FOR EACH whiteboard solution code cell, THE code cell SHALL execute to completion on MLX_Runtime without raising an exception.
4. FOR EACH whiteboard solution code cell, THE code cell SHALL call `mx.eval` on any lazy MLX array before asserting on its values.
5. WHERE a whiteboard challenge specifies an expected complexity, THE paired solution cell SHALL achieve that complexity.

### Requirement 5: Complexity Analysis Faithfulness

**User Story:** As a systems-focused candidate, I want every stated FLOPs/memory/latency formula to be backed by a measured benchmark in the same notebook, so that claims are empirically grounded.

#### Acceptance Criteria

1. FOR EACH Complexity Analysis cell, THE Transform_Script SHALL insert a paired benchmark code cell in the same Notebook.
2. FOR EACH benchmark code cell, THE code SHALL use `time.perf_counter` and `mx.eval` to measure wall-clock latency.
3. FOR EACH benchmark code cell, THE code SHALL include at least 3 warmup iterations before the timed loop.
4. FOR EACH Complexity Analysis cell, THE cell SHALL declare FLOPs, memory, measured latency, and bottleneck classification (`compute`, `memory`, or `memory-bandwidth`).
5. FOR EACH Complexity Analysis cell in a Systems-tier notebook, THE cell SHALL include measurements at no fewer than two distinct sequence lengths or batch sizes.

### Requirement 6: Narrative Preservation (Additive-Only Transformation)

**User Story:** As a beginner returning to a previously-studied notebook, I want the original beginner-friendly content to be unchanged, so that my prior mental model still applies.

#### Acceptance Criteria

1. FOR EACH Notebook, THE Transform_Script SHALL insert new cells only via `mcp_jupyter_editor_ipynb_insert_cell`.
2. FOR EACH Notebook, THE Transform_Script SHALL NOT delete or replace any pre-existing cell unless the cell is explicitly listed in the notebook's manifest override.
3. FOR EACH cell listed in a notebook's `preserve_sections` manifest entry, THE post-transform cell content SHALL be byte-identical to the pre-transform content.
4. FOR EACH Notebook, THE Transform_Script SHALL anchor insertions by searching for markdown headings and inserting after the matched cell index.
5. FOR EACH Notebook, THE Transform_Script SHALL insert a `---` separator cell between preserved prose and any new interview-layer cell.

### Requirement 7: Execution Correctness on MLX Apple Silicon

**User Story:** As a reader executing the notebook, I want every cell (original and new) to run successfully on Apple Silicon with MLX, so that I can reproduce all results locally.

#### Acceptance Criteria

1. FOR EACH transformed Notebook, THE Verification_Harness SHALL execute the notebook end-to-end via `jupyter nbconvert --to notebook --execute --inplace`.
2. IF any cell raises an exception during notebook execution, THEN THE Verification_Harness SHALL exit with a non-zero status code and report the failing cell index.
3. FOR EACH new code cell, THE cell SHALL execute to completion using MLX_Runtime without raising an exception.
4. FOR EACH new code cell, THE cell SHALL NOT import `torch`, `tensorflow`, or `jax`.
5. FOR EACH new code cell that creates an MLX array, THE cell SHALL call `mx.eval` before reading the array's values on the host.

### Requirement 8: Existing Property Test Preservation

**User Story:** As a maintainer of the repository's test suite, I want all existing property-based tests to keep passing after every transform, so that regressions are caught immediately.

#### Acceptance Criteria

1. AFTER EACH notebook transform, THE Verification_Harness SHALL run `.venv/bin/python -m pytest tests/ -v`.
2. AFTER EACH notebook transform, THE test suite SHALL report at least 34 passing tests.
3. IF any existing test fails after a transform, THEN THE pipeline SHALL abort before committing to Git.
4. THE test suite execution SHALL be non-interactive (`pytest` invoked without watch mode).

### Requirement 9: New Property Tests for Interview-Grade Invariants

**User Story:** As a quality engineer, I want property-based tests for the interview-layer invariants, so that bijection, verifiability, and faithfulness are continuously enforced.

#### Acceptance Criteria

1. THE repository SHALL include a property test verifying bijection between Interview Question cells and Question_Bank records.
2. THE repository SHALL include a property test verifying that every Whiteboard Challenge cell is followed by a solution code cell containing at least one assertion.
3. THE repository SHALL include a property test verifying that every Complexity Analysis cell is paired with a benchmark code cell in the same Notebook.
4. THE repository SHALL include a property test verifying that every Question_Bank record's `difficulty`, `roles`, and `id` fields conform to the schema in Requirement 3.
5. FOR EACH new property test, THE test SHALL run at least 100 iterations using `hypothesis` strategies.

### Requirement 10: Idempotent Transform Scripts

**User Story:** As an operator running the transformation pipeline, I want each transform script to be safe to re-run, so that I can recover from partial failures without corrupting the notebook.

#### Acceptance Criteria

1. FOR EACH Transform_Script, RUNNING the script a second time on an already-transformed Notebook SHALL produce a Notebook byte-identical to the first run's output.
2. FOR EACH Transform_Script, THE script SHALL detect existing interview-layer cells by their emoji prefix or cell id and skip re-insertion.
3. FOR EACH Transform_Script, THE script SHALL reconcile the Question_Bank slice (`nb{NN}-q*`) with the Notebook's Interview Question cells on each run.
4. IF a Transform_Script fails mid-execution, THEN re-running the script SHALL complete the transform without duplicating cells.

### Requirement 11: Python-Script-Only Execution

**User Story:** As a reviewer auditing the transformation pipeline, I want all code execution to be traceable to a file in `scripts/transform/`, so that no ad-hoc inline commands can silently alter the notebooks.

#### Acceptance Criteria

1. THE transformation pipeline SHALL expose a single entry point at `scripts/transform/run_pipeline.py`.
2. FOR EACH notebook transform operation, THE operation SHALL be implemented as a Python module under `scripts/transform/`.
3. THE transformation pipeline SHALL NOT accept inline Python code passed via `-c` flags on the command line.
4. FOR EACH shell invocation in the pipeline's documentation, THE invocation SHALL reference a `.py` file under `scripts/transform/`.

### Requirement 12: Parallelism and Concurrency

**User Story:** As an operator with 20 notebooks to transform, I want the pipeline to process multiple notebooks concurrently while avoiding data races, so that total wall-clock time is minimized.

#### Acceptance Criteria

1. THE pipeline SHALL support running up to 10 notebook transforms concurrently.
2. WHERE multiple transforms run concurrently, THE Question_Bank writes SHALL be coordinated via a file lock.
3. WHERE multiple transforms run concurrently, EACH transform SHALL write only to its own `nb{NN}-q*` key-space within the Question_Bank.
4. IF a concurrent Question_Bank write conflict is detected, THEN THE transform SHALL retry with exponential backoff.

### Requirement 13: Post-Transform Pedagogy Review on All Notebooks

**User Story:** As a curriculum owner, I want every transformed notebook to be re-scored by the Pedagogy_Reviewer sub-agent, so that no notebook falls below the Interview_Ready_Bar unnoticed.

#### Acceptance Criteria

1. AFTER EACH Notebook is transformed, THE pipeline SHALL invoke the Pedagogy_Reviewer sub-agent on that Notebook.
2. THE Pedagogy_Reviewer SHALL be invoked on all 20 transformed Notebooks (not a sample).
3. THE Pedagogy_Reviewer invocations SHALL be executed using parallel sub-agents respecting the concurrency cap in Requirement 12.1.
4. FOR EACH Pedagogy_Reviewer invocation, THE resulting review SHALL be persisted under `.kiro/specs/interview-grade-notebooks/reviews/nb{NN}.md`.
5. IF any Notebook's total Quality_Rubric score is below 48 or any single dimension scores below 6, THEN THE pipeline SHALL flag the Notebook for rework.

### Requirement 14: Interview-Ready Quality Threshold

**User Story:** As a hiring-candidate using the curriculum, I want every notebook to meet a consistent quality bar, so that I can trust any notebook to be interview-ready.

#### Acceptance Criteria

1. FOR EACH transformed Notebook, THE Notebook SHALL achieve a Quality_Rubric total score of at least 48 out of 60.
2. FOR EACH transformed Notebook, NO single Quality_Rubric dimension score SHALL be below 6 out of 10.
3. FOR EACH Quality_Rubric evaluation, THE six dimensions SHALL be Technical Depth, Interview Breadth, Systems Rigor, Frontier Currency, Production Anchoring, and Failure-Mode Literacy.

### Requirement 15: Complete Delivery (No Partial Rollout)

**User Story:** As a curriculum owner, I want all 20 notebooks delivered together as a coherent set, so that readers encounter a uniform experience across the curriculum.

#### Acceptance Criteria

1. THE feature SHALL be considered complete only when all 20 Notebooks (00 through 19) have been transformed and verified.
2. IF any Notebook fails verification, THEN THE feature SHALL remain incomplete until that Notebook passes.
3. THE pipeline SHALL NOT mark the feature as shipped while any Notebook remains in an untransformed or failing state.

### Requirement 16: Git Push After Each Notebook Transform

**User Story:** As a collaborator tracking progress, I want each successfully-transformed notebook to be pushed to GitHub immediately, so that progress is visible and recoverable.

#### Acceptance Criteria

1. AFTER EACH Notebook passes verification, THE pipeline SHALL stage the Notebook and related artifacts, commit with a conventional-commits message, and push to `origin main`.
2. IF verification fails for a Notebook, THEN THE pipeline SHALL NOT commit or push that Notebook's changes.
3. FOR EACH commit produced by the pipeline, THE commit message SHALL reference the Notebook number and the transform stage.
4. THE `added_in` field of each new Question_Bank record SHALL contain the Git SHA of the commit that introduced it.

### Requirement 17: Role-Filtered Reading Paths via Question Bank

**User Story:** As a candidate preparing for a specific role, I want to filter the question bank by role to generate a targeted study path, so that I can focus on role-relevant questions.

#### Acceptance Criteria

1. THE Question_Bank SHALL be the single source of role-filtered reading paths.
2. WHERE a consumer filters Question_Bank records by `roles` containing a given role, THE filtered set SHALL contain only records annotated with that role.
3. THE feature SHALL NOT produce a separate standalone reading-paths markdown file.

### Requirement 18: Out-of-Scope Items

**User Story:** As a spec reviewer, I want explicit out-of-scope declarations, so that scope creep is prevented and follow-up work is clearly identified.

#### Acceptance Criteria

1. THE feature SHALL NOT include a mock-interview script; a mock-interview script is deferred to a follow-up feature.
2. THE feature SHALL NOT produce a standalone `READING_PATHS.md` file; reading paths are expressed via Question_Bank filters per Requirement 17.
3. THE feature SHALL NOT introduce new notebooks beyond the existing 20.
4. THE feature SHALL NOT re-architect the `utils/` package.
5. THE feature SHALL NOT introduce PyTorch, TensorFlow, or JAX dependencies.

### Requirement 19: Dependency Constraints

**User Story:** As a developer setting up the project, I want the dependency surface to be explicit, so that I can reproduce the environment on Apple Silicon.

#### Acceptance Criteria

1. THE feature SHALL depend on MLX at version 0.18 or greater.
2. THE feature SHALL depend on `nbconvert` for notebook execution.
3. THE feature SHALL depend on `pytest` and `hypothesis` for the existing and new property-based test suites.
4. THE feature SHALL use the MCP Jupyter notebook tools (`mcp_jupyter_editor_ipynb_*`) for all programmatic cell edits.
5. THE feature SHALL invoke Python exclusively via `.venv/bin/python`.

### Requirement 20: Cell Template Consistency

**User Story:** As a reader scanning across notebooks, I want each cell stratum to have a visually distinct and consistent template, so that I can quickly identify cell types.

#### Acceptance Criteria

1. FOR EACH Interview Question cell, THE cell SHALL begin with the emoji prefix `🎯` followed by the question id.
2. FOR EACH Whiteboard Challenge cell, THE cell SHALL begin with the emoji prefix `🧑‍💻` followed by the challenge title.
3. FOR EACH Complexity Analysis cell, THE cell SHALL begin with the emoji prefix `📐` followed by the operation name.
4. FOR EACH Production Context cell, THE cell SHALL begin with the emoji prefix `🏭`.
5. FOR EACH Frontier Context cell, THE cell SHALL begin with the emoji prefix `🔭` followed by the topic.
6. FOR EACH Debugging & Failures cell, THE cell SHALL begin with the emoji prefix `🛠️` followed by the symptom.
7. FOR EACH transformed Notebook, THE Notebook SHALL contain exactly one end-of-notebook Interview Question Index cell titled `📋 Interview Question Index`.

### Requirement 21: Error Recovery

**User Story:** As an operator running the pipeline, I want clear error handling for common failure modes, so that I can diagnose and recover quickly.

#### Acceptance Criteria

1. IF `nbconvert --execute` fails for a Notebook, THEN THE Verification_Harness SHALL print the failing cell index and exit with a non-zero status.
2. IF the Question_Bank bijection check detects drift, THEN THE Verification_Harness SHALL list the unmatched ids on both sides and exit with a non-zero status.
3. IF a whiteboard solution cell raises an MLX out-of-memory error, THEN the error SHALL be caught and the peak memory usage SHALL be printed before exiting with a non-zero status.
4. IF the pipeline aborts mid-run, THEN re-invoking `run_pipeline.py` with the same arguments SHALL resume without duplicating cells (leveraging Requirement 10).
