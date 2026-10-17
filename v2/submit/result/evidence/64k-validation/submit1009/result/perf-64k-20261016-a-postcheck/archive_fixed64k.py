#!/usr/bin/env python3
"""Freeze completed 64K control and review evidence without editing original raw."""
import fcntl
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/data/wangzy/submit1009')
RUN='perf-64k-20261016-a'
RAW=ROOT/'result'/RUN
POST=ROOT/'result/perf-64k-20261016-a-postcheck'
REVIEW=ROOT/'result/perf-64k-20261016-a-independent-review-b'
SOURCE=ROOT/'02-测试与结果/performance-validation-20261016'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

assert (RAW/'finished').exists()
state=json.loads((ROOT/'result'/f'{RUN}-state.json').read_text())
assert state['phase']=='performance-validation-complete'
assert not (Path('/proc')/str(state['pid'])).exists()
review=json.loads((REVIEW/'review.json').read_text())
assert review['status']=='THREE_ROUND_64K1K_RAW_PERFORMANCE_REVIEW_PASS'
lease=(ROOT/'result/gpu45-owner.lock').open('a+')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
container=json.loads(subprocess.check_output(['docker','inspect','qwen38-submit1009-perf45-20261016']))[0]
assert container['Config']['Labels']['qwen38.submit1009.performance']=='20261016'
assert not container['State']['Running']
POST.mkdir(exist_ok=False)
for name in (f'{RUN}-job.log',f'{RUN}-launch.json'):
    shutil.copy2(ROOT/'logs'/name,POST/name)
shutil.copy2(ROOT/'result'/f'{RUN}-state.json',POST/'live-state-at-postcheck.json')
for name in ('review_fixed64k.py','review_fixed64k_b.py','archive_fixed64k.py'):
    shutil.copy2(SOURCE/name,POST/name)
shutil.copytree(ROOT/'result/perf-64k-20261016-a-independent-review',POST/'initial-review-failed',ignore=shutil.ignore_patterns('__pycache__'))
runtime=ROOT/'experiments/performance-validation-20261016'
shutil.copy2(runtime/'runtime-manifest.json',POST/'runtime-manifest.json')
shutil.copytree(runtime/'runtime-dependencies/aiter',POST/'measured-runtime-aiter-configs')
shutil.copy2(ROOT/'result/perf-sharegpt-20261016-a-state.json',POST/'sharegpt-overlap-state.json')
gpu=subprocess.run(['/opt/dtk/bin/rocm-smi','-d','4','5','--showuse','--showmeminfo','vram','--json'],capture_output=True,text=True,check=True)
proof=dict(status='64K_COMPLETE_PAIR45_CONTAINER_STOPPED_LOCK_FREE',utc=datetime.now(timezone.utc).isoformat(),
    host_pid=state['pid'],host_pid_exists=False,physical_pair=[4,5],tp=2,container_running=False,
    pair45_lock='FREE',gpu_probe=json.loads(gpu.stdout),accuracy='NOT_RUN',
    other_pair='6,7 remains exclusively held by existing ShareGPT matrix; untouched',
    default_patch_sha256=sha(ROOT/'patch/sitecustomize.py'),
    stock_dat_sha256=sha(ROOT/'config/library_gpu4_int8tuned_stock_clean/GemmGeneralityAdd.dat'))
(POST/'release-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
repair=dict(status='INDEPENDENT_REVIEW_B_PASS_RAW_UNCHANGED',
    original_review='initial-review-failed',initial_failure='Byte comparison incorrectly appended a trailing newline absent from benchmark atomic_json',
    fix='Only benchmark_json serialization drops the artificial newline; validated_summary serialization keeps its existing newline',
    reviewer_a_sha256=sha(SOURCE/'review_fixed64k.py'),reviewer_b_sha256=sha(SOURCE/'review_fixed64k_b.py'),
    gpu_rerun=False,raw_modified=False,model_failure=False)
(POST/'reviewer-repair.json').write_text(json.dumps(repair,indent=2)+'\n')
manifest=json.loads((RAW/'complete-sha256.json').read_text())
manifest[str((RAW/'complete-sha256.json').relative_to(ROOT))]=sha(RAW/'complete-sha256.json')
for parent in (POST,REVIEW):
    for p in sorted(parent.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:manifest[str(p.relative_to(ROOT))]=sha(p)
records={rel:dict(sha256=digest,bytes=(ROOT/rel).stat().st_size) for rel,digest in manifest.items()}
large=[rel for rel,x in records.items() if x['bytes']>=16*1024*1024]
inventory=dict(status='64K_POSTCHECK_COMPLETE',files=records,large_remote_only=large,
    raw_manifest_sha256=sha(RAW/'complete-sha256.json'),independent_review_sha256=sha(REVIEW/'review.json'))
(POST/'archive-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
print(json.dumps(dict(status=inventory['status'],files=len(records),large_remote_only=large,
    mean_gain_percent=review['comparison']['gain_percent'],review_sha256=inventory['independent_review_sha256'])))
