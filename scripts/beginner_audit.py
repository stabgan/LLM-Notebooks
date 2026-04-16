#!/usr/bin/env python3
"""
Beginner-friendliness audit for notebooks.
Checks for things that would confuse a first-time learner.
Usage: .venv/bin/python scripts/beginner_audit.py
"""

import json
import glob
import re

def load_nb(path):
    with open(path) as f:
        return json.load(f)

def src(cell):
    return ''.join(cell.get('source', []))

def audit(path):
    nb = load_nb(path)
    cells = nb.get('cells', [])
    issues = []
    name = path.split('/')[-1]
    
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown')
    all_code = ' '.join(src(c) for c in cells if c.get('cell_type') == 'code')
    
    # 1. Does the notebook start with a "What you'll learn" or learning objectives?
    first_md = src(cells[0]) if cells and cells[0].get('cell_type') == 'markdown' else ''
    if 'learn' not in first_md.lower() and 'objective' not in first_md.lower() and 'cover' not in first_md.lower():
        issues.append('NO_OBJECTIVES: First cell lacks clear learning objectives')
    
    # 2. Are there "plain English" explanations before technical content?
    # Check if markdown cells use simple language patterns
    simple_patterns = ['in plain english', 'in simple terms', 'think of it', 'imagine', 
                       'analogy', 'like a', 'for example', "let's say", 'picture this',
                       'in other words', 'simply put']
    simple_count = sum(1 for p in simple_patterns if p in all_md.lower())
    if simple_count < 2:
        issues.append(f'FEW_ANALOGIES: Only {simple_count} simple-language patterns found (need more analogies)')
    
    # 3. Are there "What just happened?" recap cells after complex code?
    recap_patterns = ['what just happened', 'what did we just', 'let\'s recap', 'to summarize',
                      'key takeaway', 'in summary', 'what we learned']
    recap_count = sum(1 for p in recap_patterns if p in all_md.lower())
    if recap_count < 1 and len(cells) > 15:
        issues.append('NO_RECAPS: No "what just happened" recap cells after complex code')
    
    # 4. Are tensor shapes explained inline?
    # Good: "# shape: (batch, seq_len, d_model)"
    shape_comments = len(re.findall(r'#\s*shape:', all_code, re.IGNORECASE))
    code_cells = sum(1 for c in cells if c.get('cell_type') == 'code')
    if code_cells > 5 and shape_comments < 3:
        issues.append(f'FEW_SHAPE_COMMENTS: Only {shape_comments} shape comments in {code_cells} code cells')
    
    # 5. Are there "Why?" explanations (not just "What")
    why_count = all_md.lower().count('why ')
    if why_count < 3 and len(cells) > 10:
        issues.append(f'FEW_WHY: Only {why_count} "why" explanations - beginners need to know WHY, not just WHAT')
    
    # 6. Are there prerequisite checks / "you should know" sections?
    has_prereq = 'prerequisite' in all_md.lower() or 'you should know' in all_md.lower() or 'before you start' in all_md.lower()
    if not has_prereq and '00_' not in name:
        issues.append('NO_PREREQS: No prerequisite knowledge listed')
    
    # 7. Are there "Try it yourself" or exercise prompts?
    exercise_patterns = ['try it', 'exercise', 'your turn', 'challenge', 'experiment', 'modify']
    exercise_count = sum(1 for p in exercise_patterns if p in all_md.lower())
    if exercise_count < 1 and len(cells) > 10:
        issues.append('NO_EXERCISES: No "try it yourself" prompts for active learning')
    
    # 8. Are error messages explained?
    # Check if there are try/except blocks with explanations
    has_error_handling = 'try:' in all_code or 'except' in all_code
    if not has_error_handling and code_cells > 5:
        issues.append('NO_ERROR_HANDLING: No try/except blocks - beginners will hit errors with no guidance')
    
    # 9. Are there "Common mistakes" or "Gotcha" sections?
    gotcha_patterns = ['common mistake', 'gotcha', 'watch out', 'be careful', 'don\'t forget',
                       'easy to forget', 'common error', 'pitfall']
    gotcha_count = sum(1 for p in gotcha_patterns if p in all_md.lower())
    # Already have pitfall emoji, so this is less critical
    
    # 10. Is there a glossary or key terms section?
    has_glossary = 'glossary' in all_md.lower() or 'key terms' in all_md.lower() or 'vocabulary' in all_md.lower()
    
    # 11. Are imports explained?
    import_cells = [i for i, c in enumerate(cells) if c.get('cell_type') == 'code' and 'import' in src(c)]
    for idx in import_cells:
        code = src(cells[idx])
        if code.count('import') > 3:
            # Check if there's a preceding markdown explaining the imports
            if idx > 0 and cells[idx-1].get('cell_type') == 'markdown':
                prev_md = src(cells[idx-1])
                if 'import' not in prev_md.lower() and 'librar' not in prev_md.lower() and 'package' not in prev_md.lower():
                    issues.append(f'UNEXPLAINED_IMPORTS: Cell {idx} has {code.count("import")} imports without explanation')
    
    # 12. Are there "Next steps" at the end?
    last_cells = [src(c) for c in cells[-3:] if c.get('cell_type') == 'markdown']
    last_md = ' '.join(last_cells)
    if 'next' not in last_md.lower() and len(cells) > 10:
        issues.append('NO_NEXT_STEPS: No "next steps" or "what\'s next" at the end')
    
    # 13. Check for jargon without definition
    jargon = ['tensor', 'gradient', 'backpropagation', 'epoch', 'batch', 'loss function',
              'optimizer', 'learning rate', 'embedding', 'attention', 'transformer',
              'encoder', 'decoder', 'tokenizer', 'softmax', 'cross-entropy']
    undefined_jargon = []
    for term in jargon:
        if term in all_md.lower():
            # Check if it's defined (has a colon or dash after it, or "is a", "means")
            pattern = f'{term}[^.]*(?:is a|means|refers to|:| —| --)'
            if not re.search(pattern, all_md.lower()):
                # Only flag if it appears in the first half of the notebook
                first_half = ' '.join(src(c) for c in cells[:len(cells)//2] if c.get('cell_type') == 'markdown')
                if term in first_half.lower():
                    undefined_jargon.append(term)
    
    if undefined_jargon and len(undefined_jargon) > 2:
        issues.append(f'UNDEFINED_JARGON: Terms used without definition: {", ".join(undefined_jargon[:5])}')
    
    return issues

if __name__ == '__main__':
    notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    print(f"Beginner-friendliness audit of {len(notebooks)} notebooks\n")
    
    total = 0
    by_type = {}
    
    for path in notebooks:
        issues = audit(path)
        name = path.split('/')[-1]
        total += len(issues)
        
        status = '✅' if len(issues) <= 1 else ('⚠️' if len(issues) <= 3 else '❌')
        print(f"{status} {name:<45} {len(issues)} issues")
        for issue in issues:
            tag = issue.split(':')[0]
            by_type[tag] = by_type.get(tag, 0) + 1
            print(f"   └─ {issue}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total} beginner-friendliness issues")
    print(f"\nBy category:")
    for tag, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}× {tag}")
