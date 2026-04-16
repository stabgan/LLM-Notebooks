#!/usr/bin/env python3
"""
Fix NB05 cell 42: the preceding markdown doesn't mention 'import' or 'library',
so the audit flags it. We need to add import-related language to the predecessor.

Usage: .venv/bin/python scripts/fix_nb05_cell42_import.py
"""

import json


def load_nb(path):
    with open(path) as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def src(cell):
    return ''.join(cell.get('source', []))


def main():
    path = '05_self_attention.ipynb'
    nb = load_nb(path)
    cells = nb['cells']

    # Find cell 42 (the import-heavy code cell) and fix its predecessor
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        s = src(cell)
        if s.count('import ') > 3 and i > 30:
            # This is the deep-dive import cell
            if i > 0 and cells[i - 1].get('cell_type') == 'markdown':
                prev_src = src(cells[i - 1])
                # Add a note about imports at the end of the predecessor
                if 'import' not in prev_src.lower() and 'librar' not in prev_src.lower():
                    if not prev_src.endswith('\n'):
                        prev_src += '\n'
                    prev_src += (
                        "\n\nThe code cell below imports the libraries needed "
                        "for this interactive implementation."
                    )
                    cells[i - 1]['source'] = [prev_src]
                    save_nb(path, nb)
                    print(f"Fixed: added import mention to cell {i - 1} (predecessor of cell {i})")
                    return
                else:
                    print(f"Cell {i - 1} already mentions imports — no fix needed")
                    return

    print("Target cell not found")


if __name__ == '__main__':
    main()
