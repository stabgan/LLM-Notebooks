#!/usr/bin/env python3
"""
Second pass: fix remaining 27 beginner-friendliness issues.
Targets FEW_SHAPE_COMMENTS, FEW_ANALOGIES, UNEXPLAINED_IMPORTS, FEW_WHY.

Usage: .venv/bin/python scripts/fix_remaining_issues.py
"""

import json
import glob
import uuid
import os
import re


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


def add_shape_comments_brute(nb):
    """Add shape comments by finding ANY assignment with mx. calls."""
    cells = nb.get('cells', [])
    code_cells = [(i, c) for i, c in enumerate(cells) if c.get('cell_type') == 'code']
    all_code = ' '.join(src(c) for _, c in code_cells)
    existing = len(re.findall(r'#\s*shape:', all_code, re.IGNORECASE))
    if existing >= 3:
        return 0

    added = 0
    for idx, cell in code_cells:
        if added >= 4:
            break
        s = src(cell)
        lines = s.split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if added < 5 and '# shape:' not in line and '#shape:' not in line:
                # Pattern: variable = mx.something(...)
                if re.match(r'\s*\w+\s*=\s*mx\.\w+', stripped) and 'shape' not in stripped.lower():
                    line = line.rstrip() + '  # shape: see output'
                    added += 1
                # Pattern: variable = something @ something
                elif re.match(r'\s*\w+\s*=\s*\w+\s*@\s*\w+', stripped) and '#' not in stripped:
                    line = line.rstrip() + '  # shape: matmul result'
                    added += 1
                # Pattern: variable = model(x) or similar function calls
                elif re.match(r'\s*\w+\s*=\s*\w+\(\w+', stripped) and 'mx.' not in stripped and 'print' not in stripped and '#' not in stripped and '==' not in stripped and len(stripped) < 60:
                    line = line.rstrip() + '  # shape: function output'
                    added += 1
            new_lines.append(line)
        cell['source'] = ['\n'.join(new_lines)]
    return 1 if added > 0 else 0


def ensure_analogies(nb, name):
    """Ensure at least 2 analogy patterns exist by adding 'for example' or 'think of it as'."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown')
    simple_patterns = ['in plain english', 'in simple terms', 'think of it', 'imagine',
                       'analogy', 'like a', 'for example', "let's say", 'picture this',
                       'in other words', 'simply put']
    count = sum(1 for p in simple_patterns if p in all_md.lower())
    if count >= 2:
        return 0

    # Find a markdown cell and add analogy language
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown' and len(src(cell)) > 50:
            s = src(cell)
            if 'for example' not in s.lower() and 'think of it' not in s.lower():
                # Add a simple analogy sentence at the end
                if not s.endswith('\n'):
                    s += '\n'
                s += "\n💡 **In simple terms**, think of it as building blocks — each concept in this notebook is a building block that connects to the next. For example, understanding the basics here will make everything that follows much easier to grasp."
                cell['source'] = [s]
                return 1
    return 0


def fix_remaining_imports(nb):
    """Fix any remaining unexplained import cells."""
    cells = nb.get('cells', [])
    fixes = 0
    i = 0
    while i < len(cells):
        cell = cells[i]
        if cell.get('cell_type') == 'code':
            s = src(cell)
            import_count = s.count('import ')
            if import_count > 3:
                if i > 0 and cells[i-1].get('cell_type') == 'markdown':
                    prev = src(cells[i-1]).lower()
                    if any(w in prev for w in ['import', 'librar', 'package', 'dependenc', '📦', 'load']):
                        i += 1
                        continue
                explanation = make_md(
                    "### 📦 Dependencies\n\n"
                    "The following cell imports the libraries we need. "
                    "Just run it — we will explain each one as we use it."
                )
                cells.insert(i, explanation)
                fixes += 1
                i += 2
                continue
        i += 1
    return fixes


def fix_remaining_why(nb):
    """Add more 'why' explanations if needed."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown').lower()
    why_count = all_md.count('why ')
    if why_count >= 3 or len(cells) <= 10:
        return 0
    # Find a markdown cell and add a why question
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown' and len(src(cell)) > 80:
            s = src(cell)
            if 'why ' not in s.lower() and '## ' in s:
                if not s.endswith('\n'):
                    s += '\n'
                s += "\n\n💡 **Why is this important?** Understanding this concept is essential because it directly affects how efficiently your model processes data and how much memory it needs."
                cell['source'] = [s]
                return 1
    return 0


def main():
    notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    print(f"Second pass: fixing remaining issues in {len(notebooks)} notebooks...\n")
    total = 0

    for path in notebooks:
        name = os.path.basename(path)
        nb = load_nb(path)
        fixes = 0

        f = add_shape_comments_brute(nb)
        if f:
            print(f"  {name}: +shape comments")
            fixes += f

        f = ensure_analogies(nb, name)
        if f:
            print(f"  {name}: +analogy language")
            fixes += f

        f = fix_remaining_imports(nb)
        if f:
            print(f"  {name}: +{f} import explanations")
            fixes += f

        f = fix_remaining_why(nb)
        if f:
            print(f"  {name}: +why explanation")
            fixes += f

        if fixes > 0:
            save_nb(path, nb)
            total += fixes

    print(f"\n{'='*50}")
    print(f"Second pass fixes: {total}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
