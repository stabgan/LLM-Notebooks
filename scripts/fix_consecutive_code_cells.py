#!/usr/bin/env python3
"""
Fix consecutive code cells by inserting brief markdown explanations between them.
Usage: .venv/bin/python scripts/fix_consecutive_code_cells.py <notebook_path>
"""

import json
import sys
import uuid

def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_notebook(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

def get_cell_source(cell):
    return ''.join(cell.get('source', []))

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [source]
    }

def extract_context_from_code(code_source):
    """Try to extract a meaningful description from code comments or function names."""
    lines = code_source.strip().split('\n')
    
    # Look for a leading comment
    for line in lines[:5]:
        stripped = line.strip()
        if stripped.startswith('# ') and len(stripped) > 10:
            return stripped[2:]
    
    # Look for class/function definitions
    for line in lines[:10]:
        stripped = line.strip()
        if stripped.startswith('class '):
            name = stripped.split('(')[0].replace('class ', '')
            return f"Define the `{name}` class"
        if stripped.startswith('def '):
            name = stripped.split('(')[0].replace('def ', '')
            return f"Implement the `{name}` function"
    
    # Look for print statements that describe what's happening
    for line in lines[:5]:
        stripped = line.strip()
        if stripped.startswith('print(') and "'" in stripped:
            # Extract the string being printed
            try:
                msg = stripped.split("'")[1]
                if len(msg) > 10:
                    return msg[:80]
            except IndexError:
                pass
    
    return None

def fix_notebook(path, dry_run=False):
    """Insert markdown cells between consecutive code cells."""
    nb = load_notebook(path)
    cells = nb.get('cells', [])
    
    new_cells = []
    insertions = 0
    
    for i, cell in enumerate(cells):
        if i > 0 and cell.get('cell_type') == 'code' and cells[i-1].get('cell_type') == 'code':
            # Try to generate a meaningful description
            code_src = get_cell_source(cell)
            context = extract_context_from_code(code_src)
            
            if context:
                md_text = f"The next cell continues the implementation:\n\n**{context}**"
            else:
                # Generic but still helpful
                prev_src = get_cell_source(cells[i-1])
                prev_context = extract_context_from_code(prev_src)
                if prev_context:
                    md_text = f"Building on the previous step, the next cell extends the implementation."
                else:
                    md_text = "The following cell continues the implementation from above."
            
            new_cells.append(make_markdown_cell(md_text))
            insertions += 1
        
        new_cells.append(cell)
    
    if insertions > 0 and not dry_run:
        nb['cells'] = new_cells
        save_notebook(path, nb)
        print(f"  ✅ Inserted {insertions} markdown cells into {path}")
    elif insertions > 0:
        print(f"  [DRY RUN] Would insert {insertions} markdown cells into {path}")
    else:
        print(f"  ✅ No consecutive code cells found in {path}")
    
    return insertions

if __name__ == '__main__':
    import glob
    
    dry_run = '--dry-run' in sys.argv
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        paths = [sys.argv[1]]
    else:
        paths = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    
    total = 0
    for path in paths:
        total += fix_notebook(path, dry_run=dry_run)
    
    print(f"\nTotal insertions: {total}")
