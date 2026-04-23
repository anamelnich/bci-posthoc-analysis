import json
from pathlib import Path

path = Path('notebooks/posthoc_analysis.ipynb')
nb = json.loads(path.read_text())
updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'Expected: 16 subjects' in src:
            print('FOUND cell source with outdated expectation:')
            print(src)
            src_new = src.replace(
                'Expected: 16 subjects × 2 groups × 2 sessions × 3 trial_types = 192 rows',
                'Expected: 16 subjects × 2 sessions × 3 trial_types = 96 rows'
            )
            src_new = src_new.replace(
                'Expected: 16 subjects × 1 group × 2 sessions × 3 trial_types = 96 rows',
                'Expected: 16 subjects × 2 sessions × 3 trial_types = 96 rows'
            )
            if src_new != src:
                cell['source'] = [src_new]
                updated = True

if updated:
    path.write_text(json.dumps(nb, indent=1))
    print('Updated notebook cells.')
else:
    print('No change needed.')
