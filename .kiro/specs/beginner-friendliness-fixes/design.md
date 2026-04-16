# Design: Beginner-Friendliness Fixes

## Approach
A single comprehensive Python script (`scripts/fix_all_beginner_issues.py`) will iterate over all 20 notebooks and apply targeted fixes for each issue category. The script will be idempotent — running it multiple times produces the same result.

## Fix Strategies by Category

### NO_ERROR_HANDLING
- Insert a `try/except` block in the first substantive code cell of each affected notebook
- The block should demonstrate handling a common error (e.g., import failure, shape mismatch)

### FEW_SHAPE_COMMENTS
- Scan code cells for tensor operations (`mx.`, `@`, `.reshape`, `.transpose`)
- Add `# shape: (...)` comments to the first 3+ such operations found

### UNEXPLAINED_IMPORTS
- Before any code cell with 4+ imports, insert a markdown cell: "### 📦 Imports\nHere we load the libraries needed for this section..."

### FEW_ANALOGIES
- Add a markdown cell near the top of each affected notebook with a plain-English analogy for the notebook's core concept

### NO_OBJECTIVES
- Modify the first markdown cell to include "**What you'll learn:**" or "**What we'll cover:**" with bullet points

### NO_RECAPS
- After the longest code cell in each affected notebook, insert a "### 🔍 What Just Happened?" markdown cell

### FEW_WHY
- Add "**Why?**" or "**Why does this matter?**" explanations in existing markdown cells

### NO_NEXT_STEPS
- Append a "## ➡️ What's Next?" markdown cell at the end of each affected notebook

### NO_EXERCISES
- Add a "## 🧪 Try It Yourself" section with 2-3 beginner exercises

### UNDEFINED_JARGON
- Add inline definitions for technical terms used without explanation

## Testing
- Run `scripts/beginner_audit.py` after fixes — target: 0 issues
- Run `pytest tests/ -v` — all 34 tests must pass
