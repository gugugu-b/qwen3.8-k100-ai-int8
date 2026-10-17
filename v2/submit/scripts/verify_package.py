#!/usr/bin/env python3
"""Verify immutable package and, optionally, the full external rocBLAS library."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def verify_library(path):
    expected=json.loads((ROOT/'config/library-sha256.json').read_text())
    for rel,digest in expected.items():
        p=path/rel
        if not p.is_file() or not p.resolve().is_relative_to(path.resolve()):
            raise ValueError(f'Missing library file: {p}')
        if sha(p)!=digest:raise ValueError(f'Library SHA mismatch: {p}')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual != set(expected):raise ValueError(f'Unexpected library files: {actual-set(expected)}')
    return len(expected)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',action='store_true')
    p.add_argument('--library',type=Path,default=ROOT/'config/library_gpu4_int8tuned_stock_clean')
    a=p.parse_args()
    m=json.loads((ROOT/'manifest/package-sha256.json').read_text())
    for rel,digest in m.items():
        f=ROOT/rel
        if not f.is_file() or not f.resolve().is_relative_to(ROOT):raise SystemExit(f'Missing/unsafe file: {rel}')
        if sha(f)!=digest:raise SystemExit(f'Package SHA mismatch: {rel}')
    for rel,entry in json.loads((ROOT/'result/summary/final-performance-report-20261016.json').read_text())['source_inventory']['files'].items():
        if sha(ROOT/'result/summary'/rel)!=entry['sha256']:raise SystemExit(f'Evidence mismatch: {rel}')
    assert sha(ROOT/'patch/sitecustomize.py')=='da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610'
    assert sha(ROOT/'patch/native_bf16_ar_guard.py')=='55485f3994417b012fd76f3cf423b03b33fd90d52e2c73d98d0fc9315ab0b5aa'
    report=dict(status='PACKAGE_INTEGRITY_PASS',files=len(m),runtime='NOT_CHECKED',accuracy='NOT_EVALUATED_BY_INTEGRITY_CHECK',submission_ready=False)
    if a.runtime:
        try:report.update(runtime='LIBRARY_SHA256_PASS',library_files=verify_library(a.library))
        except (ValueError,OSError) as e:raise SystemExit(str(e))
    print(json.dumps(report,ensure_ascii=False,indent=2))
