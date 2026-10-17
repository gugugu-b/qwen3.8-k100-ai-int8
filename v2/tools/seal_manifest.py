#!/usr/bin/env python3
"""Seal current source/evidence files. Run again after an intentional package edit."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'submit'
target=root/'manifest/package-sha256.json'
manifest={}
for p in sorted(root.rglob('*')):
    if not p.is_file() or p==target or p.name=='.DS_Store' or '__pycache__' in p.parts:continue
    rel=str(p.relative_to(root))
    if rel.startswith('result/runs/'):continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    manifest[rel]=h.hexdigest()
target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(f'Sealed {len(manifest)} files')
