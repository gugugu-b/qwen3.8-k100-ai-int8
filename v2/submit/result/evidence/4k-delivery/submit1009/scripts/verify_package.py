#!/usr/bin/env python3
"""CPU-only complete manifest validation; never imports GPU packages."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'config/package-sha256.json').read_text())
for rel, digest in manifest.items():
    path = root / rel
    assert path.is_file() and path.resolve().is_relative_to(root), rel
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    assert h.hexdigest() == digest, rel
print(json.dumps({'status': 'PACKAGE_SHA256_PASS', 'files': len(manifest)}))
