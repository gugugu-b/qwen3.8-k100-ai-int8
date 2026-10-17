#!/usr/bin/env python3
"""Prepare a private, auditable physical-device assertion for reviewer HCU0,1."""
import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
runs = (ROOT / 'result/runs').resolve()
assert out.is_relative_to(runs) and not out.exists(), 'Use a new directory under result/runs'
source = ROOT / 'patch'
manifest = json.loads((source / 'native-ar-manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
old = (source / 'native_bf16_ar_guard.py').read_text()
needle = "assert os.environ['HIP_VISIBLE_DEVICES'] == '6,7'"
assert old.count(needle) == 1
new = old.replace(needle, "assert os.environ['HIP_VISIBLE_DEVICES'] == '0,1'")
out.mkdir(parents=True)
for name in manifest['files']:
    shutil.copy2(source / name, out / name)
(out / 'native_bf16_ar_guard.py').write_text(new)
digest = hashlib.sha256(new.encode()).hexdigest()
manifest['files']['native_bf16_ar_guard.py'] = digest
(out / 'native-ar-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(out / 'physical-device.diff').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True),
    fromfile='patch/native_bf16_ar_guard.py', tofile='review01/native_bf16_ar_guard.py')))
record = {'status': 'PRIVATE_REVIEW01_PATCH_PREPARED', 'physical_devices': [0, 1],
          'original_guard_sha256': hashlib.sha256(old.encode()).hexdigest(),
          'review_guard_sha256': digest, 'change': 'Only physical device assertion 6,7 -> 0,1',
          'original_package_modified': False, 'files': manifest['files']}
(out / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
