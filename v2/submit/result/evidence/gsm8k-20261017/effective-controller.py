#!/usr/bin/env python3
"""Four owned TP2 instances. Candidate-only diagnostics, immutable runs, no accuracy."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

PACKAGE=Path('/data/wangzy/v2/submit')
SOURCE=PACKAGE/'execution/official-alignment-20261017-c'
CONTROL=Path('/data/wangzy/submit1009/result/community-official-20261017-c')
LIBRARY=PACKAGE/'config/library_gpu4_int8tuned_stock_clean'
MODEL=Path('/data/model/Qwen3.8-27B-Channel-INT8-w8a8')
DATASET=Path('/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json')
IMAGE='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
RUN='community-official-20261017-c'
SITE_SHA='da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def run(cmd,**kw):return subprocess.run(cmd,check=True,timeout=kw.pop('timeout',120),**kw)
def output(cmd):return subprocess.check_output(cmd,text=True,timeout=60)
def smi(pair):return json.loads(output(['/opt/dtk/bin/rocm-smi','-d',*pair.split(','),'--showuse','--showmeminfo','vram','--json']))
def verify_owned(name,pair):
 x=json.loads(output(['docker','inspect',name]))[0]
 assert x['Image']==IMAGE and x['Config']['Labels'].get('qwen38.community.run')==RUN
 assert x['Config']['Labels'].get('qwen38.community.pair')==pair
 assert f'HIP_VISIBLE_DEVICES={pair}' in x['Config']['Env']
 return x

def make_runtime(pair):
 target=CONTROL/('runtime-p'+pair.replace(',',''))
 target.mkdir(exist_ok=False)
 patch=target/'patch'
 shutil.copytree(PACKAGE/'patch',patch,ignore=shutil.ignore_patterns('baseline-v1','runtime','__pycache__','*.pyc'))
 guard=patch/'native_bf16_ar_guard.py';original=guard.read_text()
 needle="assert os.environ['HIP_VISIBLE_DEVICES'] == '6,7'";assert original.count(needle)==1
 guard.write_text(original.replace(needle,f"assert os.environ['HIP_VISIBLE_DEVICES'] == '{pair}'"))
 m=json.loads((patch/'native-ar-manifest.json').read_text());m['files']['native_bf16_ar_guard.py']=sha(guard)
 m['physical_pair']=pair;m['adaptation']='Physical pair assertion only; no arithmetic/routing change'
 write(patch/'native-ar-manifest.json',m)
 probe=target/'probe';probe.mkdir();shutil.copy2(guard,probe/guard.name)
 text=(SOURCE/'probe_template.py').read_text().replace("== '0,1'",f"== '{pair}'").replace('physical_device=rank+0',f'physical_device=rank+{pair[0]}')
 (probe/'probe.py').write_text(text)
 write(target/'identity.json',dict(site_sha256=sha(patch/'sitecustomize.py'),guard_sha256=sha(guard),pair=pair,original_guard_sha256=sha(PACKAGE/'patch/native_bf16_ar_guard.py')))
 assert sha(patch/'sitecustomize.py')==SITE_SHA
 return target

def create_container(pair,port,target):
 key=pair.replace(',','');name=f'qwen38-community-v2-{key}-gsm8k-20261017-e'
 assert subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,name
 cmd=['docker','run','-d','--name',name,'--label',f'qwen38.community.run={RUN}','--label',f'qwen38.community.pair={pair}',
 '--network','host','--ipc','host','--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--cap-add','SYS_PTRACE','--security-opt','seccomp=unconfined','--security-opt','label=disable']
 env=dict(HIP_VISIBLE_DEVICES=pair,GPU_PAIR=pair,PHYSICAL_PAIR=pair,PORT=str(port),NCCL_PORT=str(port+1),
 PATCH_DIR=str(target/'patch'),MODEL_PATH=str(MODEL),DATASET_PATH=str(DATASET),VARIANT='candidate',
 RUN_ID=f'{RUN}-p{key}-server',MAX_WAIT='1800',MAX_GATE_WAIT='5',TVM_FFI_CACHE_DIR='/run/qwen38-tvm-ffi',
 PYTHONDONTWRITEBYTECODE='1',Q38_EXTERNAL_PAIR_LOCK='1')
 for k,v in env.items():cmd+=['-e',k+'='+v]
 binds=[(PACKAGE,PACKAGE,False),(CONTROL,CONTROL,False),(MODEL,MODEL,True),(DATASET,DATASET,True),
 (Path('/opt/hyhal'),Path('/opt/hyhal'),True),(LIBRARY,Path('/run/qwen38-library'),True),
 (PACKAGE/'patch/runtime/tvm-ffi',Path('/run/qwen38-tvm-ffi'),True),
 (PACKAGE/'patch/runtime/direct_register_custom_op.py',Path('/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py'),True)]
 for p in (PACKAGE/'config/aiter').glob('*/*.json'):
  binds.append((p,Path('/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs')/p.parent.name/p.name,True))
 for src,dst,ro in binds:cmd+=['-v',f'{src}:{dst}'+(':ro' if ro else '')]
 cmd += [IMAGE,'sleep','infinity']
 cid=output(cmd).strip();obj=verify_owned(name,pair)
 write(target/'docker-run-command.json',cmd);write(target/'container-inspect.json',obj)
 # Capture installed package versions without injecting sitecustomize.
 with (target/'pip-freeze.txt').open('x') as f:run(['docker','exec',name,'python3','-I','-m','pip','freeze'],stdout=f,timeout=120)
 return name

def gate(name,pair,target,port):
 out=target/'gate';out.mkdir()
 cmd=['docker','exec']
 for k,v in dict(HIP_VISIBLE_DEVICES=pair,CUDA_VISIBLE_DEVICES=pair,SGLANG_USE_1STAGE_ALLREDUCE='1',SGLANG_USE_QUICK_CUSTOM_ALLREDUCE='false',
 PYTHONPATH=str(target/'probe'),AR_PROBE_OUTPUT=str(out),HSA_FORCE_FINE_GRAIN_PCIE='1',PYTHONDONTWRITEBYTECODE='1').items():cmd+=['-e',k+'='+v]
 cmd +=[name,'bash','-lc',f'source /opt/dtk/env.sh; exec python3 -B -m torch.distributed.run --nproc-per-node 2 --master-addr 127.0.0.1 --master-port {port+2} {target}/probe/probe.py']
 with (out/'probe.log').open('x') as f:run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 ranks=[json.loads((out/f'rank-{i}/summary.json').read_text()) for i in range(2)]
 for rank,r in enumerate(ranks):
  assert r['status']=='NUMERICAL_GRAPH_PASS' and r['cases']==9 and r['excluded_routes']==10
  assert r['physical_device']==int(pair.split(',')[rank])
  assert r['poisoned_full_write_checks']==r['public_eager_checks']==r['changed_graph_checks']==27
 write(out/'review.json',dict(status='RUNTIME_PAIR_NUMERICAL_GRAPH_PASS',pair=pair,checks=162,exclusions=20,accuracy=False,actual_model_graph_gate='REVIEW_REQUIRED',ranks=ranks))

def bench(name,pair,scene,c,tag):
 run_id=f'{RUN}-p{pair.replace(",","")}-{tag}-{scene}-c{c}'
 with (CONTROL/f'p{pair.replace(",","")}/{tag}-{scene}-c{c}.log').open('x') as f:
  run(['docker','exec','-e','RUN_ID='+run_id,name,'bash',str(PACKAGE/'scripts/benchmark_reference.sh'),scene,str(c)],stdout=f,stderr=subprocess.STDOUT,timeout=8*3600)
 directory=PACKAGE/f'result/runs/{run_id}/{scene}-c{c}'
 report=json.loads((directory/'reference_performance_review.json').read_text())
 cfg=json.loads((directory/'config.json').read_text())
 assert report['performance_status']=='PASS' and report['measured_rounds']==3
 rounds=[json.loads(p.read_text()) for p in sorted(directory.glob('round-*/summary.json'))]
 limits=(10,80) if scene=='4k' else (30,100)
 diagnostic=dict(challenge_p95=all(s['ttft_s']['p95']<limits[0] and s['tpot_ms']['p95']<limits[1] for s in rounds),
 tutorial_p95=all(s['ttft_s']['p95']<(3 if scene=='4k' else 15) and s['tpot_ms']['p95']<50 for s in rounds))
 return dict(run_id=run_id,scene=scene,c=c,path=str(directory),validation=report,config=cfg,provisional_SLA_by_statistic=diagnostic,official_SLA_max=None)

def worker(w,stage):
 pair=w['pair'];key=pair.replace(',','');statepath=CONTROL/f'p{key}/state.json'
 state=dict(pair=pair,pid=os.getpid(),utc=now(),stage=stage,status='PREPARING',scenes=[],accuracy='NOT_RUN',parallel_instances=4)
 name=None;lease=None
 try:
  lease=Path(f'/data/wangzy/submit1009/result/gpu{key}-owner.lock').open('a+')
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Do not rewrite the existing lock file payload; actual flock is the authority.
  before=smi(pair)
  for card,v in before.items():assert float(v['HCU use (%)'])==0 and float(v['vram Total Used Memory (MiB)'])<=16,(card,v)
  write(statepath,state);write(CONTROL/f'p{key}/gpu-before.json',before)
  for p in [w['port'],w['port']+1,w['port']+2]:
   with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',p))
  target=make_runtime(pair);name=create_container(pair,w['port'],target)
  state.update(status='PAIR_GATE',container=name);write(statepath,state)
  gate(name,pair,target,w['port'])
  state.update(status='MODEL_STARTING');write(statepath,state)
  with (CONTROL/f'p{key}/startup.log').open('x') as f:
   run(['docker','exec',name,'bash',str(PACKAGE/'scripts/start_server.sh')],stdout=f,stderr=subprocess.STDOUT,timeout=2000)
  serverlog=PACKAGE/f'logs/{RUN}-p{key}-server/server.log'
  text=serverlog.read_text()
  assert text.count('[Native BF16 AR] Installed')>=2
  info=json.loads((serverlog.parent/'server-info.json').read_text());args=info.get('server_args',info)
  assert args['tp_size']==2 and args['max_running_requests']==32 and args['cuda_graph_max_bs']==32
  assert args['disable_cuda_graph'] is False and args['disable_custom_all_reduce'] is False and args['custom_all_reduce_backend']=='native'
  write(CONTROL/f'p{key}/server-info.json',info)
  for item in w['matrix']:
   state.update(status='BENCHMARKING',current=item,utc=now());write(statepath,state)
   result=bench(name,pair,item['scene'],item['c'],'matrix');state['scenes'].append(result);write(statepath,state)
  # Keep no GPU resident after this phase; phase two starts only after matrix review.
  state.update(status='MATRIX_COMPLETE',utc=now());write(statepath,state)
 except BaseException as e:
  state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),utc=now());write(statepath,state)
 finally:
  if name:
   try:
    verify_owned(name,pair)
    with (CONTROL/f'p{key}/container-final-top.txt').open('x') as f:subprocess.run(['docker','top',name,'-eo','pid,ppid,args'],stdout=f,stderr=subprocess.STDOUT,timeout=30)
    run(['docker','stop','--time','30',name],stdout=subprocess.DEVNULL,timeout=90)
    write(CONTROL/f'p{key}/container-stopped.json',verify_owned(name,pair))
   except BaseException as e:state['cleanup_error']=repr(e);state['status']='CLEANUP_FAILED'
  if lease:
   fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
  try:state['gpu_after']=smi(pair)
  except BaseException as e:state['gpu_observation_error']=repr(e)
  state.update(finished_utc=now(),lock_released=True);write(statepath,state)
  return state

def prepare():
 assert not CONTROL.exists(),'Inspect existing control; refuse duplicate execution'
 CONTROL.mkdir(parents=True)
 write(CONTROL/'execution-plan.json',json.loads((SOURCE/'execution-plan.json').read_text()))
 run([sys.executable,'-I',str(PACKAGE/'scripts/verify_package.py')],stdout=(CONTROL/'package-check.log').open('x'))
 if not LIBRARY.exists():run([sys.executable,'-B',str(PACKAGE/'scripts/prepare_assets.py'),'--library-source','/data/wangzy/submit1009/config/library_gpu4_int8tuned_stock_clean'],stdout=(CONTROL/'prepare-assets.log').open('x'),timeout=600)
 run([sys.executable,'-I',str(PACKAGE/'scripts/verify_package.py'),'--runtime'],stdout=(CONTROL/'runtime-check.log').open('x'),timeout=600)
 assert output(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip()==IMAGE
 assert sha(DATASET)=='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
 env=dict(utc=now(),hostname=socket.gethostname(),model_source='hygon/Qwen3.8-27B-Channel-INT8-w8a8 per on-disk model README; official contest asset permission pending',
 cpu=output(['lscpu']),memory=Path('/proc/meminfo').read_text(),kernel=output(['uname','-a']),driver=Path('/sys/module/hydcu/version').read_text() if Path('/sys/module/hydcu/version').exists() else None,
 disk=output(['df','-h','/data']),other_containers=output(['docker','ps','--format','{{.Names}}\t{{.Status}}']),gpu_before=smi('0,1,2,3,4,5,6,7'))
 write(CONTROL/'environment.json',env)
 # Hash both deployed and original model before GPU measurements; IO is not overlapped with benchmarks.
 assets={}
 for folder in [MODEL,Path('/data/model/Qwen3.8-27B')]:
  files={}
  for p in sorted(folder.iterdir()):
   if p.is_file():files[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
  assets[str(folder)]=files
  write(CONTROL/'model-sha256-progress.json',assets)
 write(CONTROL/'model-sha256.json',assets)
 write(CONTROL/'preparation-finished.json',dict(utc=now(),status='ASSETS_READY_NO_GPU_WORK_YET',source_sha256={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--matrix',action='store_true');a=p.parse_args();assert a.matrix
 prepare()
 plan=json.loads((SOURCE/'execution-plan.json').read_text())
 write(CONTROL/'controller-state.json',dict(status='MATRIX_RUNNING',pid=os.getpid(),utc=now(),accuracy='NOT_RUN'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(worker,w,'random_matrix') for w in plan['workers']]
  results=[]
  for f in concurrent.futures.as_completed(futures):
   result=f.result();results.append(result)
   write(CONTROL/'completed-workers.json',results)
 status='MATRIX_COMPLETE' if all(s['status']=='MATRIX_COMPLETE' for s in results) else 'MATRIX_HAS_FAILURES'
 write(CONTROL/'controller-state.json',dict(status=status,pid=os.getpid(),utc=now(),workers=results,accuracy='NOT_RUN',next='CPU review then SLA diagnostics; do not label official scoring'))
 write(CONTROL/'finished.json',dict(status=status,utc=now()))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if CONTROL.exists():write(CONTROL/'controller-failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
  raise
