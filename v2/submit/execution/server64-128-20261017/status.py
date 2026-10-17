#!/usr/bin/env python3
"""Compact read-only remote status; does not query the inference API or GPU."""
import subprocess
from pathlib import Path
workspace=Path(__file__).resolve().parents[2]
code=r'''import json,time,re
from pathlib import Path
r=Path('/data/wangzy/v2/submit');control=Path('/data/wangzy/v2/retest-server64-128-20261017-a')
out={'utc_epoch':time.time(),'workers':[]}
for name in ['controller-state.json','controller-failure.json','finished.json']:
 p=control/name
 if p.exists():
  d=json.loads(p.read_text());out[name]={k:d[k] for k in ['status','utc','error','pid'] if k in d}
for key,c in [('45',64),('67',128)]:
 p=control/f'p{key}/state.json';v={'pair':key,'client_c':c}
 if p.exists():
  s=json.loads(p.read_text());v.update({k:s[k] for k in ['status','utc','error','server_run_id','cleanup_status','lock_released'] if k in s})
 d=r/f'result/runs/high-concurrency-20261017-a-p{key}-server{c}-4k-c{c}/4k-c{c}'
 if d.exists():
  if (d/'progress.json').exists():v['progress']=json.loads((d/'progress.json').read_text())
  v['phase_directories']=[p.name for p in sorted(d.iterdir()) if p.is_dir()]
  v['completed_round_tps']=[json.loads(p.read_text())['output_throughput_tps'] for p in sorted(d.glob('round-*/summary.json'))]
 f=control/f'runtime-p{key}/running-metrics.jsonl'
 if f.exists() and f.stat().st_size:
  with f.open('rb') as h:
   h.seek(max(0,f.stat().st_size-4*1024*1024));rows=h.read().splitlines()
  last=json.loads(rows[-1]);v['metrics_latest_utc']=last.get('utc')
  v['current_running_gauges']=[{'name':x.split('{')[0],'value':float(x.rsplit(' ',1)[1])} for x in last.get('text','').splitlines() if x.startswith('sglang:num_running_reqs{') or x.startswith('sglang:num_queue_reqs{')]
 if v.get('status') in ('MODEL_STARTING','FAILED','CLEANUP_FAILED'):
  log=r/f'logs/high-concurrency-20261017-a-p{key}-server/server.log'
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-8000));lines=f.read().decode(errors='replace').splitlines()
   v['server_tail']=[re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',x) for x in lines[-5:]]
 out['workers'].append(v)
print(json.dumps(out,ensure_ascii=False,separators=(',',':')))
'''
subprocess.run(['python3',str(workspace/'05-工具/remote_exec.py'),'--host','192.2.111.66',"python3 - <<'HC_STATUS_REMOTE'\n"+code+"\nHC_STATUS_REMOTE"],check=True)
