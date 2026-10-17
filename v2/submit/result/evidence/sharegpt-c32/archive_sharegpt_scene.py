#!/usr/bin/env python3
"""Snapshot completed ShareGPT stages while later scenes continue unchanged."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--concurrency',type=int,choices=[32,16,64,4,1],required=True)
p.add_argument('--root',type=Path,default=Path('/data/wangzy/submit1009'))
a=p.parse_args()
ROOT=a.root.resolve(); RUN='perf-sharegpt-20261016-a'; OUT=ROOT/'result'/RUN
POST=ROOT/'result'/f'{RUN}-c{a.concurrency}-postcheck'
assert not POST.exists(), 'Never overwrite a snapshot or restart a completed scene'
for kind in ('v1','candidate','v1-return'):
    meta=json.loads((OUT/'stages'/f'c{a.concurrency}-{kind}.json').read_text())
    assert meta['status']=='COMPLETE' and meta['physical_pair']==[6,7]
producer=OUT/f'review-c{a.concurrency}/comparison.json'
produced=json.loads(producer.read_text())
assert produced['status']=='SHAREGPT_PERFORMANCE_RAW_REVIEW_PASS'
assert str(a.concurrency) in produced['scenarios']
manifest_bytes=(OUT/'artifact-sha256.json').read_bytes()
manifest=json.loads(manifest_bytes)
# A snapshot of the last completed-stage manifest excludes the current stage's
# live files. Every listed source must still exactly match its frozen digest.
for rel,digest in manifest.items():
    path=ROOT/rel
    assert path.resolve().is_relative_to(ROOT) and path.is_file()
    assert sha(path)==digest, rel
POST.mkdir()
SNAP=POST/'submit1009'
for rel,digest in manifest.items():
    target=SNAP/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/rel,target)
    assert sha(target)==digest,rel
snapshot_manifest=SNAP/'result'/RUN/'artifact-sha256.json'
snapshot_manifest.write_bytes(manifest_bytes)
shutil.copy2(producer,POST/'producer-comparison.json')
shutil.copy2(Path(__file__),POST/'archive_sharegpt_scene.py')
shutil.copy2(ROOT/'result'/f'{RUN}-state.json',POST/'current-queue-state.json')
shutil.copy2(ROOT/'experiments/performance-validation-20261016/runtime-manifest.json',POST/'runtime-manifest.json')
review_source=SNAP/'result'/RUN/'source/review_sharegpt_matrix.py'
assert review_source.is_file()
with (POST/'independent-review.log').open('x') as f:
    subprocess.run([sys.executable,'-B',str(review_source),'--root',str(SNAP),'--run',RUN,
        '--allow-partial','--out',str(POST/'remote-independent-review')],check=True,stdout=f,stderr=subprocess.STDOUT)
independent=POST/'remote-independent-review/comparison.json'
review=json.loads(independent.read_text())
assert review['scenarios'][str(a.concurrency)]==produced['scenarios'][str(a.concurrency)]
proof=dict(status='COMPLETED_SCENE_SNAPSHOT_INDEPENDENT_REVIEW_PASS',utc=datetime.now(timezone.utc).isoformat(),
    concurrency=a.concurrency,run=RUN,physical_pair=[6,7],tp=2,
    snapshot_manifest_sha256=sha(snapshot_manifest),
    producer_manifest_sha256=produced['manifest_sha256'],
    producer_comparison_sha256=sha(producer),independent_comparison_sha256=sha(independent),
    full_report_byte_identical_to_producer=producer.read_bytes()==independent.read_bytes(),
    scene_result_identical_to_producer=True,source_files=len(manifest),
    live_queue_modified=False,accuracy='NOT_RUN',formal_acceptance=False,frozen=False)
(POST/'snapshot-review.json').write_text(json.dumps(proof,indent=2)+'\n')
files={str(path.relative_to(POST)):dict(bytes=path.stat().st_size,sha256=sha(path))
       for path in sorted(POST.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
inventory=dict(status='SHAREGPT_SCENE_SNAPSHOT_COMPLETE',concurrency=a.concurrency,
    files=files,large_remote_only=[rel for rel,item in files.items() if item['bytes']>=16*1024*1024],
    snapshot_proof=proof)
(POST/'archive-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(dict(status=inventory['status'],concurrency=a.concurrency,files=len(files),
    large_remote_only=inventory['large_remote_only'],gain=review['scenarios'][str(a.concurrency)]['candidate_gain_percent'],
    producer_and_independent_byte_identical=proof['full_report_byte_identical_to_producer'])))
