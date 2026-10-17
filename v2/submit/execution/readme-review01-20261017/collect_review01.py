#!/usr/bin/env python3
"""Collect completed README execution and prove API/device/patch identity."""
from pathlib import Path
import subprocess,json,re,hashlib
W=Path(__file__).resolve().parents[2]
P=W/'v2/submit';out=P/'result/evidence/readme-review01-20261017'
host='192.2.111.66';remote='/data/wangzy/v2'
query=r'''from pathlib import Path
import json,subprocess,fcntl,datetime
base=Path('/data/wangzy/v2'); e=base/'readme-env-20261017'
id=(e/'review01-id.txt').read_text().strip()
log=(e/'review01-smoke.log').read_text()
assert 'REVIEW01_README_SMOKE_PASS '+id in log
r=subprocess.run(['docker','inspect','qwen38-v2-review01'],capture_output=True)
assert r.returncode!=0,'Review container is still present'
with open('/data/wangzy/submit1009/result/gpu01-owner.lock','a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.run(['/opt/dtk/bin/rocm-smi','-d','0','1','--showuse','--showmeminfo','vram'],text=True,capture_output=True,check=True)
print(json.dumps({'status':'REVIEW01_README_SMOKE_AND_RELEASE_PASS','run_id':id,'container_absent':True,'lock_free':True,'gpu_after':gpu.stdout,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver':Path('/sys/module/hycu/version').read_text().strip()}))'''
r=subprocess.run(['python3',str(W/'05-工具/remote_exec.py'),'--host',host,"python3 - <<'REVIEW_EVIDENCE'\n"+query+"\nREVIEW_EVIDENCE"],text=True,capture_output=True,check=True)
state=json.loads(r.stdout);rid=state['run_id']
raw=out/'server'
paths=['readme-env-20261017',f'submit/logs/{rid}-server',f'submit/result/runs/{rid}-runtime',f'submit/result/runs/{rid}-assets']
if not raw.exists():
 subprocess.run(['python3',str(W/'05-工具/remote_transfer.py'),'download',remote,str(raw),*paths,'--host',host,'--dereference-download'],check=True)
serverlog=(raw/f'submit/logs/{rid}-server/server.log').read_text()
assert sorted(set(map(int,re.findall(r'\[Native BF16 AR\] Eligible rank=(\d)',serverlog))))==[0,1]
api=json.loads((raw/f'submit/logs/{rid}-server/server-info.json').read_text())
args=api.get('server_args',api);assert args['tp_size']==2 and args['max_running_requests']==32 and args['cuda_graph_max_bs']==32
private=raw/f'submit/result/runs/{rid}-runtime/patch'
prep=json.loads((private/'preparation.json').read_text())
assert prep['review_guard_sha256']==hashlib.sha256((private/'native_bf16_ar_guard.py').read_bytes()).hexdigest()
assets=json.loads((raw/f'submit/result/runs/{rid}-assets/assets.json').read_text());assert assets['status']=='EXTERNAL_ASSET_IDENTITY_PASS'
state.update(native_guard_eligible_ranks=[0,1],service_limit=32,graph_limit=32,assets_status=assets['status'],review_guard_sha256=prep['review_guard_sha256'],source_image_id='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad',benchmark_run=False)
(out/'runtime-review.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(state,ensure_ascii=False,indent=2))
