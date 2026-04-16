#!/usr/bin/env python3
"""
Notebook improvement script - analyzes and fixes pedagogical issues.
Run with: .venv/bin/python scripts/improve_notebooks.py <notebook_path>
"""

import json
import sys
import os

def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_notebook(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

def get_cell_source(cell):
    return ''.join(cell.get('source', []))

def make_markdown_cell(source, cell_id=None):
    import uuid
    return {
        "cell_type": "markdown",
        "id": cell_id or str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [source]
    }

def analyze_notebook(path):
    """Analyze a notebook and return issues found."""
    nb = load_notebook(path)
    cells = nb.get('cells', [])
    issues = []
    
    basename = os.path.basename(path)
    
    # Check 1: Does it have a history section?
    has_history = any('📜 History' in get_cell_source(c) for c in cells)
    if not has_history:
        issues.append(('missing_history', 'No 📜 History & Alternatives section'))
    
    # Check 2: Are all 4 emoji markers present?
    all_source = ' '.join(get_cell_source(c) for c in cells)
    for emoji, name in [('💡', 'insight'), ('⚡', 'performance'), ('🎯', 'interview'), ('⚠️', 'pitfall')]:
        if emoji not in all_source:
            issues.append(('missing_emoji', f'Missing {emoji} ({name}) marker'))
    
    # Check 3: Are there long code cells that should be split?
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            src = get_cell_source(cell)
            lines = src.split('\n')
            if len(lines) > 80:
                issues.append(('long_cell', f'Cell {i} has {len(lines)} lines - consider splitting'))
    
    # Check 4: Do code cells have preceding markdown explanations?
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and i > 0:
            prev = cells[i-1]
            if prev.get('cell_type') == 'code':
                issues.append(('consecutive_code', f'Cells {i-1} and {i} are consecutive code cells without explanation'))
    
    # Check 5: Are there enough visualizations?
    viz_count = sum(1 for c in cells if 'plt.show()' in get_cell_source(c) or 'plot' in get_cell_source(c).lower())
    if viz_count < 2 and len(cells) > 10:
        issues.append(('few_visualizations', f'Only {viz_count} visualization(s) found'))
    
    # Check 6: Check for MLX-only (no PyTorch/TensorFlow)
    for i, cell in enumerate(cells):
        src = get_cell_source(cell)
        if 'import torch' in src or 'import tensorflow' in src:
            issues.append(('non_mlx', f'Cell {i} imports PyTorch/TensorFlow'))
    
    # Check 7: Shape assertions present?
    has_shape_assert = any('assert' in get_cell_source(c) and 'shape' in get_cell_source(c) for c in cells)
    if not has_shape_assert and any('mx.' in get_cell_source(c) for c in cells):
        issues.append(('no_shape_asserts', 'No shape assertions found in code'))
    
    return issues, nb, cells

def report(path):
    """Print analysis report for a notebook."""
    issues, nb, cells = analyze_notebook(path)
    basename = os.path.basename(path)
    
    print(f"\n{'='*60}")
    print(f"  {basename}")
    print(f"  Cells: {len(cells)} | Issues: {len(issues)}")
    print(f"{'='*60}")
    
    if not issues:
        print("  ✅ No issues found!")
    else:
        for issue_type, desc in issues:
            icon = {'missing_history': '❌', 'missing_emoji': '⚠️', 'long_cell': '📏', 
                    'consecutive_code': '📝', 'few_visualizations': '📊', 'non_mlx': '🚫',
                    'no_shape_asserts': '🔍'}.get(issue_type, '❓')
            print(f"  {icon} [{issue_type}] {desc}")
    
    return issues

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Analyze all notebooks
        import glob
        notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
        print(f"Analyzing {len(notebooks)} notebooks...")
        
        total_issues = 0
        for nb_path in notebooks:
            issues = report(nb_path)
            total_issues += len(issues)
        
        print(f"\n{'='*60}")
        print(f"  TOTAL: {total_issues} issues across {len(notebooks)} notebooks")
        print(f"{'='*60}")
    else:
        report(sys.argv[1])
