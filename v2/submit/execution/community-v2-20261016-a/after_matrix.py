#!/usr/bin/env python3
"""Sequential continuation: review matrix, diagnose SLA, clean-runtime checks and real Demo capture."""
import concurrent.futures
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
from datetime import datetime,timezone
from pathlib import Path

PACKAGE=Path('/data/wangzy/v2/submit')
SOURCE=PACKAGE/'execution/community-v2-20261016-a'
PREVIOUS=Path('/data/wangzy/submit1009/result/community-v2-20261016-a')
NEXT=Path('/data/wangzy/submit1009/result/community-v2-followup-20261016-b')
FOLLOWUP='community-v2-followup-20261016-b'

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def load_runtime():
 # Preserve the complete effective runtime source, with a new container name;
 # the original launched matrix controller remains unchanged.
 text=(SOURCE/'controller.py').read_text()
 old="name=f'qwen38-community-v2-{key}-20261016-a'"
 new="name=f'qwen38-community-v2-{key}-followup-20261016-b'"
 assert text.count(old)==1
 text=text.replace(old,new)
 (NEXT/'effective-controller.py').write_text(text)
 mod=types.ModuleType('followup_runtime')
 exec(compile(text,str(NEXT/'effective-controller.py'),'exec'),mod.__dict__)
 mod.CONTROL=NEXT;mod.RUN=FOLLOWUP
 return mod

def demo(core,name,pair,port,target):
 out=target/'demo';out.mkdir()
 code=r'''
import json,sys,time,urllib.request
from pathlib import Path
out=Path(sys.argv[1]);port=sys.argv[2];events=[];start=time.monotonic()
payload={"model":"Qwen3.8-27B-Channel-INT8-w8a8","messages":[{"role":"user","content":"请用中文简短解释张量并行，再写一个将列表元素求和的Python函数。"}],"max_tokens":512,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
(out/'request.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
req=urllib.request.Request('http://127.0.0.1:'+port+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=240) as r, (out/'sse-events.jsonl').open('x') as log:
 for line in r:
  row={'elapsed_s':time.monotonic()-start,'raw':line.decode('utf-8')}
  log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
  if line.startswith(b'data:'):
   raw=line[5:].strip()
   if raw==b'[DONE]':break
   value=json.loads(raw)
   if 'error' in value:raise RuntimeError(value['error'])
   for choice in value.get('choices',[]):
    delta=choice.get('delta',{});text=delta.get('reasoning_content','') or delta.get('content','')
    if text:events.append({'elapsed_s':row['elapsed_s'],'text':text});print(text,end='',flush=True)
print()
assert events
(out/'capture.json').write_text(json.dumps({'status':'REAL_API_STREAM_CAPTURED','elapsed_s':time.monotonic()-start,'events':events,'request':payload,'url':'http://127.0.0.1:'+port+'/v1/chat/completions','video_render':'PENDING_TERMINAL_REPLAY_RENDER_NOT_A_GUI_SCREEN_RECORDING'},ensure_ascii=False,indent=2)+'\n')
'''
 capture=target/'capture_demo.py';capture.write_text(code)
 with (out/'transcript.txt').open('x') as f:
  core.run(['docker','exec',name,'python3','-I',str(capture),str(out),str(port)],stdout=f,stderr=subprocess.STDOUT,timeout=300)
 return dict(status='REAL_DEMO_API_CAPTURED',path=str(out),video='NEEDS_RENDER_FROM_ACTUAL_TIMED_EVENTS')

def phase_worker(w,mode):
 core=load_runtime();pair=w['pair'];key=pair.replace(',','');statepath=NEXT/f'p{key}/state.json'
 state=dict(pair=pair,status='PREPARING',mode=mode,pid=os.getpid(),utc=now(),scenes=[],accuracy='NOT_RUN');name=None;lease=None
 try:
  lease=Path(f'/data/wangzy/submit1009/result/gpu{key}-owner.lock').open('a+')
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  g=core.smi(pair)
  for card,v in g.items():assert float(v['HCU use (%)'])==0 and float(v['vram Total Used Memory (MiB)'])<=16,(card,v)
  write(statepath,state)
  for port in [w['port'],w['port']+1,w['port']+2]:
   with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
  target=core.make_runtime(pair);name=core.create_container(pair,w['port'],target)
  state.update(status='PAIR_GATE',container=name);write(statepath,state);core.gate(name,pair,target,w['port'])
  state.update(status='MODEL_STARTING');write(statepath,state)
  with (NEXT/f'p{key}/startup.log').open('x') as f:
   core.run(['docker','exec',name,'bash',str(PACKAGE/'scripts/start_server.sh')],stdout=f,stderr=subprocess.STDOUT,timeout=2000)
  info=json.loads((PACKAGE/f'logs/{FOLLOWUP}-p{key}-server/server-info.json').read_text());args=info.get('server_args',info)
  assert args['tp_size']==2 and args['max_running_requests']==args['cuda_graph_max_bs']==32
  assert args['disable_cuda_graph'] is False and args['custom_all_reduce_backend']=='native' and args['disable_custom_all_reduce'] is False
  write(NEXT/f'p{key}/server-info.json',info)
  if mode in ('sla-4k','sla-64k'):
   scene=mode[4:];previous=json.loads((PREVIOUS/f'p{key}/state.json').read_text())
   first=next(r for r in previous['scenes'] if r['scene']==scene and r['c']==1)
   state['scenes'].append(dict(first,reused_from_matrix=True));write(statepath,state)
   def measure(c):
    state.update(status='SLA_DIAGNOSTIC',current={'scene':scene,'c':c},utc=now());write(statepath,state)
    value=core.bench(name,pair,scene,c,'sla');state['scenes'].append(value);write(statepath,state);return value
   for c in [2,4,8,16,32]:
    value=measure(c)
    if not any(value['provisional_SLA_by_statistic'].values()):break
   # Fill integer points through the first jointly failing measured level.
   # Result is only the observed boundary, never a claim of a global/official maximum.
   highest=max(r['c'] for r in state['scenes'])
   for c in range(1,highest):
    if not any(r['c']==c for r in state['scenes']):measure(c)
   diagnostic={}
   for statistic in ['mean','p99']:
    passed=sorted(r['c'] for r in state['scenes'] if r['provisional_SLA_by_statistic'][statistic])
    failed=sorted(r['c'] for r in state['scenes'] if not r['provisional_SLA_by_statistic'][statistic])
    diagnostic[statistic]=dict(largest_observed_passing_concurrency=max(passed) if passed else 0,observed_failing_concurrency=failed,upper_unexplored=True)
   state.update(status='SLA_DIAGNOSTIC_COMPLETE',diagnostic=diagnostic,official_SLA_max=None,official_protocol='PENDING_TUTORIAL')
  elif mode=='clean-runtime':
   for scene in ['4k','64k']:
    state.update(status='CLEAN_RUNTIME_BENCHMARK',current={'scene':scene,'c':8},utc=now());write(statepath,state)
    state['scenes'].append(core.bench(name,pair,scene,8,'clean-runtime'));write(statepath,state)
   state.update(status='CLEAN_RUNTIME_PERFORMANCE_COMPLETE',parallel_host_activity_disclosed=True)
  elif mode=='demo':
   state.update(status='DEMO_CAPTURE');write(statepath,state)
   state['demo']=demo(core,name,pair,w['port'],target);state['status']='DEMO_CAPTURE_COMPLETE'
  write(statepath,state)
 except BaseException as e:state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),utc=now());write(statepath,state)
 finally:
  if name:
   try:
    core.verify_owned(name,pair);core.run(['docker','stop','--time','30',name],stdout=subprocess.DEVNULL,timeout=90)
    write(NEXT/f'p{key}/container-stopped.json',core.verify_owned(name,pair))
   except BaseException as e:state.update(status='CLEANUP_FAILED',cleanup_error=repr(e))
  if lease:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
  try:state['gpu_after']=core.smi(pair)
  except BaseException as e:state['gpu_observation_error']=repr(e)
  state.update(finished_utc=now(),lock_released=True);write(statepath,state)
  return state

def write_reports(results):
 matrix=[json.loads(p.read_text()) for p in sorted(PREVIOUS.glob('p*/state.json'))]
 rows=[]
 for worker in matrix:
  for point in worker['scenes']:
   rows.append(dict(pair=worker['pair'],kind='matrix',**point))
 for worker in results:
  for point in worker['scenes']:
   rows.append(dict(pair=worker['pair'],kind=worker['mode'],**point))
 report=dict(status='EXECUTED_DIAGNOSTICS_WITH_OFFICIAL_PROTOCOL_PENDING',utc=now(),rows=rows,workers=results,
 official_max_concurrency=None,accuracy='NOT_RUN_LAST',parallel_interference='Concurrent instances on the same host',
 model_assets_manifest=str(PREVIOUS/'model-sha256.json'),final_model_graph='REVIEW_REQUIRED',frozen=False)
 out=PACKAGE/'result/summary/community-v2-execution-20261016-b.json';write(out,report)
 lines=['# 社区v2缺档与后续执行结果','',f'生成时间：{now()}（UTC时间戳）。官方教程/SLA口径待确认，以下是诊断数据，不是官方成绩。',
 '', '| 类型 | 卡对 | 场景 | C | 三轮TPS均值 | 完整性 |', '|---|---|---|---:|---:|---|']
 for r in rows:
  lines.append(f"| {r['kind']} | {r['pair']} | {r['scene']} | {r['c']} | {r['validation']['mean_output_tps']:.4f} | {r['validation']['status']} |")
 lines+=['','精度未运行，排在性能/工程之后。SLA同时保存mean/P99观察边界，官方最大并发仍未知。',
 '完整命令、服务参数、原始请求、预热及正式轮见同名JSON列出的各路径。新目录干净容器复现保留并行主机条件；不作隔离显著性声明。',
 'Demo只使用实际API时间流记录，终端回放视频仍需据此渲染；不将日志或截图冒充已交视频。',
 '实际模型Graph/compiler关联、官方工具/模型准入与四项精度仍未完结；不冻结或晋级默认。']
 (out.with_suffix('.md')).write_text('\n'.join(lines)+'\n')
 core=load_runtime()
 m={}
 roots=[PREVIOUS,NEXT,PACKAGE/'result/runs']
 for root in roots:
  for p in root.rglob('*'):
   if p.is_file() and p.name!='execution-results-sha256.json':m[str(p)]=dict(bytes=p.stat().st_size,sha256=core.sha(p),local_sync=p.stat().st_size<16*1024*1024)
 write(NEXT/'execution-results-sha256.json',m)

def main():
 assert not NEXT.exists(),'Inspect previous follow-up; do not duplicate'
 NEXT.mkdir()
 write(NEXT/'controller-state.json',dict(status='WAITING_FOR_MATRIX',pid=os.getpid(),utc=now(),accuracy='NOT_RUN'))
 while not (PREVIOUS/'finished.json').exists():
  if (PREVIOUS/'controller-failure.json').exists():
   write(NEXT/'controller-state.json',dict(status='STOPPED_MATRIX_PREPARATION_FAILED',utc=now()));return
  time.sleep(30)
 result=json.loads((PREVIOUS/'finished.json').read_text())
 if result['status']!='MATRIX_COMPLETE':
  write(NEXT/'controller-state.json',dict(status='STOPPED_MATRIX_HAS_FAILURES_REVIEW_REQUIRED',matrix_status=result['status'],utc=now()));return
 # Recheck completed raw sizes, SHA identities and formal round counts without writing originals.
 core=load_runtime()
 for p in PREVIOUS.glob('p*/state.json'):
  state=json.loads(p.read_text());assert state['status']=='MATRIX_COMPLETE' and state['lock_released']
  for scene in state['scenes']:
   d=Path(scene['path']);cfg=json.loads((d/'config.json').read_text())
   assert len(list(d.glob('round-*/requests.jsonl')))==3
   for q in d.glob('round-*/requests.jsonl'):
    rows=[json.loads(s) for s in q.read_text().splitlines()];assert len(rows)==cfg['requests_per_round']
    assert all(not r['error'] and r['done'] and r['completion_tokens']==1024 and r['prompt_tokens']==cfg['input_len'] and r.get('replacement_chars',0)==0 for r in rows)
 write(NEXT/'matrix-independent-review.json',dict(status='MATRIX_REQUEST_INTEGRITY_RECHECK_PASS',utc=now(),official_scoring=False))
 write(NEXT/'controller-state.json',dict(status='SLA_AND_ENGINEERING_RUNNING',pid=os.getpid(),utc=now(),accuracy='NOT_RUN'))
 tasks=[({'pair':'0,1','port':31200},'sla-4k'),({'pair':'2,3','port':31210},'sla-64k'),
        ({'pair':'4,5','port':31220},'demo'),({'pair':'6,7','port':31230},'clean-runtime')]
 results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(phase_worker,w,mode) for w,mode in tasks]
  for f in concurrent.futures.as_completed(fs):results.append(f.result());write(NEXT/'completed-workers.json',results)
 failed=any(r['status'] in ['FAILED','CLEANUP_FAILED'] for r in results)
 write(NEXT/'controller-state.json',dict(status='FOLLOWUP_HAS_FAILURES' if failed else 'DIAGNOSTICS_COMPLETE_PENDING_OFFICIAL_PROTOCOL_AND_ACCURACY',utc=now(),workers=results,accuracy='NOT_RUN',video='ACTUAL_TIMED_CAPTURE_AVAILABLE_IF_DEMO_SUCCEEDED; RENDER_PENDING',official_scoring=False))
 write_reports(results)
 write(NEXT/'finished.json',dict(utc=now(),failures=failed,accuracy='NOT_RUN'))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if NEXT.exists():write(NEXT/'controller-failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
  raise
