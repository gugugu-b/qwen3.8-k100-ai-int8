#!/usr/bin/env python3
"""Unpack pinned offline scorer dependencies only into this experiment folder."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
dest = root/'quality-vendor'
dest.mkdir(exist_ok=True)
manifest = []
for p in sorted((root/'quality-wheels').iterdir()):
    if p.name.startswith('evalscope-'):
        manifest.append(dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        continue
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            relative = Path(name)
            if p.suffix == '.zip':
                relative = Path(*relative.parts[1:])
                if not relative.parts or relative.parts[0] != 'word2number': continue
            assert not relative.is_absolute() and '..' not in relative.parts
            if name.endswith('/'): continue
            target = dest/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    manifest.append(dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/'quality-dependencies.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Offline scorer dependencies prepared:', len(manifest))
