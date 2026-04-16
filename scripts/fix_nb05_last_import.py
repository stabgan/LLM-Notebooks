#!/usr/bin/env python3
"""
Fix the last remaining issue: NB05 cell 42 has 5+ imports without explanation.

Usage: .venv/bin/python scripts/fix_nb05_last_import.py
"""

import json
import uuid


def load_nb(path):
    with open(path) as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def src(cell):
    return ''.join(cell.get('source', []))


def make_md(text):
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [text]
    }


def main():
    path = '05_self_attention.ipynb'
    nb = load_nb(path)
    cells = nb['cells']

    # First, find ALL code cells with 4+ imports and check their predecessors
    print("Scanning NB05 for import-heavy code cells...\n")
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        s = src(cell)
        import_count = s.count('import ')
        if import_count > 3:
            prev_type = cells[i - 1].get('cell_type', 'none') if i > 0 else 'none'
            prev_text = src(cells[i - 1])[:80] if i > 0 else ''
            print(f"  Cell {i}: {import_count} imports, prev={prev_type}, prev_text='{prev_text}...'")

            # Check if predecessor already explains imports
            if i > 0 and cells[i - 1].get('cell_type') == 'markdown':
                prev_lower = src(cells[i - 1]).lower()
                keywords = ['import', 'librar', 'package', 'dependenc', 'load', 'setup', '📦']
                if any(w in prev_lower for w in keywords):
                    print(f"    -> Already explained, skipping")
                    continue

            # Insert explanation
            explanation = make_md(
                "### 📦 Deep Dive Imports\n\n"
                "The following cell re-imports the libraries needed for "
                "the interactive deep dive section. Just run it to continue."
            )
            cells.insert(i, explanation)
            print(f"    -> FIXED: inserted explanation before cell {i}")
            save_nb(path, nb)
            print(f"\nDone! Saved {path}")
            return

    print("No unfixed import cells found.")


if __name__ == '__main__':
    main()
