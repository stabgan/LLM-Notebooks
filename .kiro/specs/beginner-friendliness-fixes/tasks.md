# Tasks: Beginner-Friendliness Fixes

## Tasks

- [x] 1. Create comprehensive fix script `scripts/fix_all_beginner_issues.py`
  - [x] 1.1 Implement NO_OBJECTIVES fixer for 9 notebooks (NB00,06,07,08,09,10,11,13,14)
    - Add "What you'll learn/cover" to first markdown cell
    - _Requirements: 5.1_
  - [x] 1.2 Implement FEW_ANALOGIES fixer for 10 notebooks (NB00,06,07,08,10,11,13,14,17, and NB08 which has only 1)
    - Add plain-English analogies using patterns like "think of it as", "imagine", "like a"
    - _Requirements: 4.1_
  - [x] 1.3 Implement NO_RECAPS fixer for 7 notebooks (NB06,07,08,11,14,15,19)
    - Add "What just happened?" recap cells after complex code sections
    - _Requirements: 6.1_
  - [x] 1.4 Implement FEW_SHAPE_COMMENTS fixer for 14 notebooks
    - Add `# shape:` comments to code cells with tensor operations
    - _Requirements: 2.1_
  - [x] 1.5 Implement UNEXPLAINED_IMPORTS fixer for 13 notebooks
    - Add markdown explanation before import-heavy code cells
    - _Requirements: 3.1_
  - [x] 1.6 Implement FEW_WHY fixer for 7 notebooks (NB06,07,09,11,13,14,17)
    - Add "Why?" explanations in markdown cells
    - _Requirements: 7.1_
  - [x] 1.7 Implement NO_NEXT_STEPS fixer for 6 notebooks (NB00,10,13,14,15,16)
    - Add "What's Next?" cells at end of notebooks
    - _Requirements: 8.1_
  - [x] 1.8 Implement NO_EXERCISES fixer for 3 notebooks (NB02,03,05)
    - Add "Try It Yourself" exercise sections
    - _Requirements: 9.1_
  - [x] 1.9 Implement NO_ERROR_HANDLING fixer for 16 notebooks
    - Add try/except demonstration blocks
    - _Requirements: 1.1_
  - [x] 1.10 Implement UNDEFINED_JARGON fixer for 2 notebooks (NB00,17)
    - Add inline definitions for technical terms
    - _Requirements: 10.1_

- [x] 2. Run fix script and verify
  - [x] 2.1 Execute `scripts/fix_all_beginner_issues.py` — applied 73 fixes (first pass) + 26 fixes (second pass) + 3 targeted fixes
  - [x] 2.2 Run `scripts/beginner_audit.py` — verified **0 issues** across all 20 notebooks
  - [x] 2.3 Run `pytest tests/ -v` — verified all **34 tests pass** in 6.81s

- [x] 3. Commit and push to GitHub
