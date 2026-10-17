#!/usr/bin/env python3
"""Collect complete immutable precision evidence, verify SHA, publish exact local-protocol result."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
W=Path(__file__).resolve().parents[2];P=W/'v2/submit'
O=W/'02-测试与结果/results-gsm8k-community-20261017-e'
REMOTE='/data/wangzy/submit1009/result/community-gsm8k-20261017-e'
HOST='192.2.111.66'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def remote(code):
 return json.loads(subprocess.check_output([sys.executable,str(W/'05-工具/remote_exec.py'),'--host',HOST,"python3 - <<'Q38_PRECISION_COLLECT'\n"+code+"\nQ38_PRECISION_COLLECT"],text=True,timeout=120))
def main():
 O.mkdir(exist_ok=True)
 assert not (O/'local-collected.json').exists()
 save(O/'dispatch.json',dict(pid=os.getpid(),time=now(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 end=time.monotonic()+30*3600;previous=None
 while time.monotonic()<end:
  try:
   x=remote("from pathlib import Path\nimport json\np=Path("+repr(REMOTE)+")\nx={f:json.loads((p/f).read_text()) for f in ['state.json','gsm8k/progress.json','finished.json'] if (p/f).exists()}\nprint(json.dumps(x))")
   save(O/'live-status.json',x)
   state=x.get('state.json',{});progress=x.get('gsm8k/progress.json',{})
   signature=(state.get('status'),progress.get('completed'))
   if signature!=previous:print(now(),json.dumps(signature),flush=True);previous=signature
   (O/'执行状态.md').write_text('# GSM8K 独立TP2执行状态\n\n'+f'更新时间：{now()}。卡对0,1，端口31400。\n\n状态：{state.get("status","WAITING")}；完成 {progress.get("completed",0)}/1319。只以完整结果判定，不用部分样本精度作结论。\n')
   if 'finished.json' in x:break
  except Exception as e:print(now(),'retry',repr(e),flush=True)
  time.sleep(30)
 else:raise TimeoutError('Precision collection exceeded 30h; remote job not killed')
 inv=remote("from pathlib import Path\nimport json,hashlib\np=Path("+repr(REMOTE)+")\nx=[]\nfor f in sorted(p.rglob('*')):\n if f.is_file() and '__pycache__' not in f.parts:x.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))\nprint(json.dumps(x))")
 save(O/'archive-inventory.json',inv)
 dest=O/'archive';dest.mkdir(exist_ok=True)
 for i in range(0,len(inv),100):
  subprocess.run([sys.executable,str(W/'05-工具/remote_transfer.py'),'download',REMOTE,str(dest),*[e['path'] for e in inv[i:i+100]],'--host',HOST],check=True,timeout=300,stdout=subprocess.DEVNULL)
 for e in inv:
  f=dest/e['path'];assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256'],e
 target=P/'result/evidence/gsm8k-20261017';assert not target.exists();shutil.copytree(dest,target)
 state=json.loads((dest/'finished.json').read_text());summaryfile=dest/'gsm8k/summary.json'
 result=json.loads(summaryfile.read_text()) if summaryfile.exists() else dict(status=state['status'],error=state.get('error'),accuracy=None)
 result.update(worker_status=state['status'],evidence='result/evidence/gsm8k-20261017',
  collection='ALL_FILES_SIZE_SHA256_PASS',collected_time=now(),submission_ready=False,
  remaining_precision=['LiveCodeBench v6','RealWorldQA','MathVision Without CI'])
 save(P/'result/summary/gsm8k-community-20261017-e.json',result)
 lines=['# GSM8K 全量精度验证（独立TP2实例）','',f'归档时间：{now()}。执行状态：{state["status"]}。','',
 '卡对0,1，TP2，候选site da5b59…，native BF16 AR，服务32/Graph32、MTP保持；与其他卡对性能测试同机并行，同一实例未混入性能请求。',
 '', '使用历史EvalScope 1.6.1冻结四样例提示，GSM8K main完整1319题。temperature=0、并发32、max_tokens沿用原始配置不传；评分只读message.content。评分器先复现历史1319题，再对新原始响应独立重评分。',
 '', '本地赛事GSM8K基线97%，准入线≥95%（1319题至少1254题正确）。请求错误保留在1319题分母中；原始响应、异常、截断和历史回归题号均保留。']
 if result.get('accuracy') is not None:
  lines+=['',f'实测：**{int(result["correct"])}/1319 = {result["accuracy"]*100:.4f}%**；95%数值门槛：{"达到" if result["threshold_pass"] else "未达到"}。',
   f'请求错误 {result["errors"]}；截断 {len(result["truncated_indices"])}；含U+FFFD响应 {len(result["text_flag_indices"])}；本地协议综合状态 {result["status"]}。',
   f'历史v1同提示精度 {result["old_accuracy"]*100:.4f}%，变化 {result["delta_percentage_points"]:+.4f}pp（历史对照，非同期A/B）。']
 else:lines+=['',f'未产生完整精度成绩，原因见原始执行状态：{state.get("error",state.get("cleanup_error","未知"))}。']
 lines+=['','正式主办方精度数据版本/子集/完整脚本尚未核实，以上为可复现的本地全量验证，不能冒称已通过官方四项精度准入。LiveCodeBench v6、RealWorldQA、MathVision仍待测。',
 '', '原始记录：[逐题请求](../result/evidence/gsm8k-20261017/gsm8k/requests.jsonl)；[汇总](../result/summary/gsm8k-community-20261017-e.json)；原始响应和独立重评分记录在同一证据目录。']
 (P/'docs/GSM8K精度验证-20261017.md').write_text('\n'.join(lines)+'\n')
 envfile=P/'env_declare.yaml';env=json.loads(envfile.read_text());env['precision']={'gsm8k':result,'other_three':'NOT_RUN'}
 env['accuracy'].update(status='GSM8K_LOCAL_RESULT_OTHER_THREE_NOT_RUN',gsm8k_result='result/summary/gsm8k-community-20261017-e.json')
 envfile.write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
 checklist=P/'docs/社区要求对照与补测清单.md';text=checklist.read_text()
 text=text.replace('| GSM8K | 97.0 | ≥95.0 | 独立TP2已排队，完整1319题；见[GSM8K执行安排](GSM8K执行安排-20261017.md) |',
  '| GSM8K | 97.0 | ≥95.0 | '+(f'{result["accuracy"]*100:.4f}%，数值门槛'+('达到' if result['threshold_pass'] else '未达到') if result.get('accuracy') is not None else '执行失败，未形成成绩')+'；见[GSM8K报告](GSM8K精度验证-20261017.md) |')
 checklist.write_text(text)
 schedule=P/'docs/GSM8K执行安排-20261017.md';schedule.write_text(schedule.read_text().replace('**当前为已排队，尚无新精度成绩。**','本轮任务已结束，执行状态及完整结果见[GSM8K精度报告](GSM8K精度验证-20261017.md)。'))
 note='\n\n## GSM8K独立实例补测 '+now()+'\n\n'+('完整精度 '+f'{result["accuracy"]*100:.4f}%，95%数值门槛'+('达到' if result['threshold_pass'] else '未达到')+'。' if result.get('accuracy') is not None else '任务未形成完整成绩。')+'本地协议和原始证据见 `v2/submit/docs/GSM8K精度验证-20261017.md`。其他三项精度与最终官方协议仍未完结。\n'
 for file in [W/'BASELINE.md',W/'调优后续计划.md',P/'README.md']:
  with file.open('a') as f:f.write(note)
 # The performance collector may later add files; its own seal refreshes the manifest again.
 subprocess.run([sys.executable,str(W/'v2/tools/seal_manifest.py')],check=True)
 subprocess.run([sys.executable,'-I',str(P/'scripts/verify_package.py')],check=True)
 # Publish the same completed precision evidence into the server submission tree.
 # The immutable original control directory is retained as the source of truth.
 publish=['result/evidence/gsm8k-20261017','result/summary/gsm8k-community-20261017-e.json',
  'docs/GSM8K精度验证-20261017.md','docs/GSM8K执行安排-20261017.md',
  'docs/社区要求对照与补测清单.md','README.md','env_declare.yaml','manifest/package-sha256.json']
 subprocess.run([sys.executable,str(W/'05-工具/remote_transfer.py'),'upload','/data/wangzy/v2/submit',str(P),*publish,'--host',HOST],check=True,timeout=600,stdout=subprocess.DEVNULL)
 subprocess.run([sys.executable,str(W/'05-工具/remote_exec.py'),'--host',HOST,'python3 -I /data/wangzy/v2/submit/scripts/verify_package.py'],check=True,timeout=180)
 save(O/'local-collected.json',dict(status='PRECISION_EVIDENCE_COLLECTED',files=len(inv),result=result,time=now()))
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(O/'local-collection-failure.json',dict(error=repr(e),time=now()));raise
