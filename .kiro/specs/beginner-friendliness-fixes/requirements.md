# Requirements: Beginner-Friendliness Fixes

## Overview
Fix all 87 remaining beginner-friendliness issues across the 20-notebook LLM learning series, as identified by the automated beginner audit (`scripts/beginner_audit.py`). The goal is to make every notebook approachable for a complete beginner learning about LLMs for the first time.

## Issue Categories (from audit)

### 1. NO_ERROR_HANDLING (16 issues)
- **Requirement 1.1**: Every notebook with 5+ code cells must have at least one `try/except` block demonstrating graceful error handling, so beginners see how to handle common failures.
- **Affected notebooks**: NB01, 02, 04, 05, 06, 07, 08, 09, 10, 11, 12, 14, 15, 16, 17, 18, 19

### 2. FEW_SHAPE_COMMENTS (14 issues)
- **Requirement 2.1**: Every notebook with 5+ code cells must have at least 3 inline `# shape:` comments in code cells, helping beginners track tensor dimensions.
- **Affected notebooks**: NB00, 03, 06, 07, 08, 09, 10, 11, 12, 15, 16, 17, 18, 19

### 3. UNEXPLAINED_IMPORTS (13 issues)
- **Requirement 3.1**: Any code cell with 4+ import statements must be preceded by a markdown cell that briefly explains what libraries are being imported and why.
- **Affected notebooks**: NB03, 04, 05, 06, 07, 08, 12, 14, 15, 16, 17, 18, 19

### 4. FEW_ANALOGIES (10 issues)
- **Requirement 4.1**: Every notebook must contain at least 2 simple-language patterns (e.g., "think of it as", "imagine", "like a", "in simple terms", "for example") to make abstract concepts accessible.
- **Affected notebooks**: NB00, 06, 07, 08, 10, 11, 13, 14, 17

### 5. NO_OBJECTIVES (9 issues)
- **Requirement 5.1**: The first markdown cell of every notebook must contain clear learning objectives using words like "learn", "objective", or "cover".
- **Affected notebooks**: NB00, 06, 07, 08, 09, 10, 11, 13, 14

### 6. NO_RECAPS (7 issues)
- **Requirement 6.1**: Every notebook with 15+ cells must have at least one "what just happened" or recap cell after complex code sections.
- **Affected notebooks**: NB06, 07, 08, 11, 14, 15, 19

### 7. FEW_WHY (7 issues)
- **Requirement 7.1**: Every notebook with 10+ cells must contain at least 3 "why" explanations — beginners need to know WHY, not just WHAT.
- **Affected notebooks**: NB06, 07, 09, 11, 13, 14, 17

### 8. NO_NEXT_STEPS (6 issues)
- **Requirement 8.1**: The last 3 cells of every notebook must contain a "next" reference pointing to the following notebook.
- **Affected notebooks**: NB00, 10, 13, 14, 15, 16

### 9. NO_EXERCISES (3 issues)
- **Requirement 9.1**: Every notebook with 10+ cells must have at least one "try it yourself" or exercise prompt.
- **Affected notebooks**: NB02, 03, 05

### 10. UNDEFINED_JARGON (2 issues)
- **Requirement 10.1**: Technical terms used in the first half of a notebook must be defined before use (with "is a", "means", or "refers to" patterns).
- **Affected notebooks**: NB00, 17

## Constraints
- All fixes must be applied via Python scripts (not direct Jupyter tool calls) to avoid quoting issues
- All 34 existing property-based tests must continue to pass after fixes
- Fixes must not change any code cell logic — only add markdown cells or comments
- The automated audit (`scripts/beginner_audit.py`) must show 0 issues after all fixes
