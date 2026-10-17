#!/usr/bin/env python3
"""Collect a completed high-capacity run and verify each remote source byte."""
import hashlib
import json
import subprocess
from pathlib import Path

WORKSPACE=Path(__file__).resolve().parents[2]
LOCAL=WORKSPACE/'02-测试与结果/results-server64-128-20261017-a'
REMOTE_BASE='/data/wangzy/v2'
CONTROL_NAME='retest-server64-128-20261017-a'
HOST='192.2.111.66'
code=r'''from pathlib import Path
import hashlib,json
base=Path('/data/wangzy/v2')
control=base/'retest-server64-128-20261017-a'
finished=json.loads((control/'finished.json').read_text())
assert (control/'progress-observer-finished.json').exists(), 'Wait for CPU observer to close'
roots=[control,base/'retest-server64-128-source-20261017']
for p in sorted(control.glob('p*/state.json')):
 s=json.loads(p.read_text())
 assert s.get('lock_released') is True,(p,s)
 for item in s.get('scenes',[]):
  if item.get('path'): roots.append(Path(item['path']))
 run=s.get('server_run_id')
 if run: roots.append(base/'submit/logs'/run)
# Include success or failed server/benchmark directories, even if no scene completed.
roots += sorted((base/'submit/logs').glob('high-concurrency-20261017-a-*'))
roots += sorted((base/'submit/result/runs').glob('high-concurrency-20261017-a-*'))
paths=sorted({str(p.relative_to(base)) for p in roots if p.exists()})
paths=[p for p in paths if not any(p.startswith(q+'/') for q in paths if q!=p)]
files={}
for name in paths:
 root=base/name
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
   files[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps({'status':'COMPLETED_RUN_INVENTORY','finished':finished,'paths':paths,'files':files}))
'''
command="python3 - <<'HC_INVENTORY'\n"+code+"\nHC_INVENTORY"
v=subprocess.run(['python3',str(WORKSPACE/'05-工具/remote_exec.py'),'--host',HOST,command],text=True,capture_output=True,check=True)
manifest=json.loads(v.stdout)
LOCAL.mkdir(exist_ok=True)
output=LOCAL/'server'
assert not output.exists(),'Refusing overwrite of collected run'
subprocess.run(['python3',str(WORKSPACE/'05-工具/remote_transfer.py'),'download',REMOTE_BASE,str(output),*manifest['paths'],'--host',HOST,'--dereference-download'],check=True)
for name,d in manifest['files'].items():
 p=output/name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert p.stat().st_size==d['bytes'] and h.hexdigest()==d['sha256'],name
(LOCAL/'server-source-inventory.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(LOCAL/'collection-review.json').write_text(json.dumps({'status':'ALL_REMOTE_FILES_SIZE_SHA256_PASS','files':len(manifest['files']),'finished':manifest['finished']},ensure_ascii=False,indent=2)+'\n')
print('COLLECTED_AND_VERIFIED',len(manifest['files']))
