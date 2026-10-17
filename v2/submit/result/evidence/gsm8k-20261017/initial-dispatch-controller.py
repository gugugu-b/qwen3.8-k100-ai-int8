#!/usr/bin/env python3
"""Wait for matrix pair01 cleanup, then exclusively own a fresh TP2 GSM8K instance."""
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from datetime import datetime,timezone
P=Path('/data/wangzy/v2/submit')
SOURCE=P/'execution/gsm8k-community-20261017'
MATRIX=Path('/data/wangzy/submit1009/result/community-official-20261017-c')
CONTROL=Path('/data/wangzy/submit1009/result/community-gsm8k-20261017-e')
RUN='community-gsm8k-20261017-e'
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 assert not CONTROL.exists(),'Refuse duplicate precision run'
 CONTROL.mkdir();state=dict(status='WAITING_FOR_PAIR01_MATRIX_CLEANUP',pid=os.getpid(),utc=now(),pair='0,1',port=31400,benchmark='GSM8K',other_accuracy='NOT_RUN')
 save(CONTROL/'state.json',state)
 name=None;lease=None;core=None
 try:
  end=time.monotonic()+24*3600
  while time.monotonic()<end:
   file=MATRIX/'p01/state.json'
   if file.exists():
    a=json.loads(file.read_text())
    if a.get('lock_released'):
     assert a['status']=='MATRIX_COMPLETE',a
     break
   if (MATRIX/'controller-failure.json').exists():raise RuntimeError('Matrix preparation failed; no automatic takeover')
   time.sleep(15)
  else:raise TimeoutError('Pair01 matrix not complete after 24h')
  lease=Path('/data/wangzy/submit1009/result/gpu01-owner.lock').open('a+')
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  text=(P/'execution/official-alignment-20261017-c/controller.py').read_text()
  old="name=f'qwen38-community-v2-{key}-official-20261017-c'"
  assert text.count(old)==1
  text=text.replace(old,"name=f'qwen38-community-v2-{key}-gsm8k-20261017-e'")
  (CONTROL/'effective-controller.py').write_text(text)
  core=types.ModuleType('precision_runtime');exec(compile(text,str(CONTROL/'effective-controller.py'),'exec'),core.__dict__)
  core.CONTROL=CONTROL;core.RUN=RUN
  state.update(status='PREPARING',utc=now());save(CONTROL/'state.json',state)
  g=core.smi('0,1');save(CONTROL/'gpu-before.json',g)
  for card,v in g.items():assert float(v['HCU use (%)'])==0 and float(v['vram Total Used Memory (MiB)'])<=16,(card,v)
  for port in [31400,31401,31402]:
   with socket.socket() as s:s.bind(('127.0.0.1',port))
  target=core.make_runtime('0,1');name=core.create_container('0,1',31400,target)
  state.update(status='PAIR_GATE',container=name);save(CONTROL/'state.json',state)
  core.gate(name,'0,1',target,31400)
  # Offline dependencies stay inside this source directory; no shared pip changes.
  dependencies=json.loads((SOURCE/'quality-dependencies.json').read_text())
  for dep in dependencies:
   assert hashlib.sha256((SOURCE/'quality-wheels'/dep['file']).read_bytes()).hexdigest()==dep['sha256'],dep['file']
  save(CONTROL/'verified-quality-dependencies.json',dependencies)
  with (CONTROL/'offline-prepare.log').open('x') as f:
   core.run(['docker','exec',name,'python3','-I',str(SOURCE/'prepare_quality.py')],stdout=f,stderr=subprocess.STDOUT)
  with (CONTROL/'scorer-reproduction.log').open('x') as f:
   core.run(['docker','exec',name,'python3','-I',str(SOURCE/'evaluate.py'),'--check-reference'],stdout=f,stderr=subprocess.STDOUT,timeout=300)
  state.update(status='MODEL_STARTING');save(CONTROL/'state.json',state)
  with (CONTROL/'startup.log').open('x') as f:
   core.run(['docker','exec',name,'bash',str(P/'scripts/start_server.sh')],stdout=f,stderr=subprocess.STDOUT,timeout=2000)
  log=P/f'logs/{RUN}-p01-server/server.log';assert log.read_text().count('[Native BF16 AR] Installed')>=2
  info=json.loads((log.parent/'server-info.json').read_text());args=info.get('server_args',info)
  assert args['tp_size']==2 and args['max_running_requests']==args['cuda_graph_max_bs']==32
  assert args['disable_cuda_graph'] is False and args['custom_all_reduce_backend']=='native' and args['disable_custom_all_reduce'] is False
  save(CONTROL/'server-info.json',info)
  state.update(status='EVALUATING_GSM8K_1319',utc=now());save(CONTROL/'state.json',state)
  with (CONTROL/'evaluation.log').open('x') as f:
   core.run(['docker','exec',name,'python3','-I',str(SOURCE/'evaluate.py'),'--output',str(CONTROL/'gsm8k'),'--base-url','http://127.0.0.1:31400'],stdout=f,stderr=subprocess.STDOUT,timeout=24*3600)
  result=json.loads((CONTROL/'gsm8k/summary.json').read_text());assert result['completed']==result['requests']==1319
  state.update(status='GSM8K_COMPLETE',summary=result,utc=now());save(CONTROL/'state.json',state)
 except BaseException as e:
  state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),utc=now());save(CONTROL/'state.json',state)
 finally:
  if name:
   try:
    core.verify_owned(name,'0,1');core.run(['docker','stop','--time','30',name],stdout=subprocess.DEVNULL,timeout=90)
    save(CONTROL/'container-stopped.json',core.verify_owned(name,'0,1'))
   except BaseException as e:state.update(status='CLEANUP_FAILED',cleanup_error=repr(e))
  if lease:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
  if core:
   try:state['gpu_after']=core.smi('0,1')
   except BaseException as e:state['gpu_observation_error']=repr(e)
  state.update(finished_utc=now(),lock_released=True);save(CONTROL/'state.json',state);save(CONTROL/'finished.json',state)
if __name__=='__main__':main()
