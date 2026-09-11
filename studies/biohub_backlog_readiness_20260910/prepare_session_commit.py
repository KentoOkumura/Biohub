import subprocess
from pathlib import Path

paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().split('\0')
changed = []
for name in filter(None, paths):
    path = Path(name)
    if path.suffix not in {'.md', '.py'} or '/assets/' in name:
        continue
    content = path.read_text()
    normalized = content.rstrip() + '\n'
    if content != normalized:
        path.write_text(normalized)
        changed.append(name)
if changed:
    subprocess.run(['git', 'add', '--', *changed], check=True)
print(f'Normalized final blank lines in {len(changed)} authored files; archived source bytes preserved.')
