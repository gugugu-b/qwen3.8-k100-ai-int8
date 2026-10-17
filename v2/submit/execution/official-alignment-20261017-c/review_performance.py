#!/usr/bin/env python3
"""Independent reference-client performance audit. Text integrity is always separate."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert not a.out.exists(),'Refuse to overwrite an audit'
cfg=json.loads((a.directory/'config.json').read_text());n=cfg['input_len'];c=cfg['concurrency'];count=max(8,2*c)
assert n in [4096,65536] and cfg['output_len']==1024
assert cfg['warmup_rounds']==2 and cfg['measure_rounds']==3 and cfg['requests_per_round']==count
assert cfg['seed']==20260912
files=sorted(a.directory.glob('round-*/summary.json'));warms=sorted(a.directory.glob('warmup-*/summary.json'))
assert len(files)==3 and len(warms)==2
manifest={};flags=[];rounds=[]
def pct(v,q):
 v=sorted(v);pos=(len(v)-1)*q/100;lo=math.floor(pos);hi=math.ceil(pos)
 return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for file in warms+files:
 raw=file.parent/'requests.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()];s=json.loads(file.read_text());expected=c if s['phase']=='warmup' else count
 assert len(rows)==s['successful']==s['exact_length']==expected and s['status']=='PASS',file
 assert len({r['request_index'] for r in rows})==expected
 for r in rows:
  assert r['status_code']==200 and r['done'] and not r['error']
  assert r['prompt_tokens']==n and r['completion_tokens']==1024
  assert all(math.isfinite(r[k]) and r[k]>=0 for k in ['e2e_s','ttft_s','tpot_ms'])
  assert r['e2e_s']>=r['ttft_s'] and math.isclose(r['tpot_ms'],1000*(r['e2e_s']-r['ttft_s'])/1023,rel_tol=1e-10)
  if r.get('replacement_chars',0):flags.append(dict(phase=s['phase'],round=s['round'],index=r['request_index'],replacement_chars=r['replacement_chars']))
 tps=sum(r['completion_tokens'] for r in rows)/s['wall_time_s'];assert math.isclose(tps,s['output_throughput_tps'],rel_tol=1e-10)
 for metric in ['ttft_s','tpot_ms','e2e_s']:
  values=[r[metric] for r in rows]
  assert math.isclose(statistics.mean(values),s[metric]['mean'],rel_tol=1e-10)
  for q in [50,95,99]:assert math.isclose(pct(values,q),s[metric]['p'+str(q)],rel_tol=1e-10),file
 manifest[str(raw.relative_to(a.directory))]=digest(raw);manifest[str(file.relative_to(a.directory))]=digest(file)
 if s['phase']=='measure':rounds.append(s)
vs=[s['output_throughput_tps'] for s in rounds]
report=dict(status='PERFORMANCE_MEASUREMENTS_COMPLETE_TEXT_FLAGGED' if flags else 'PERFORMANCE_MEASUREMENTS_PASS',
 performance_status='PASS',text_integrity_status='FLAGGED_UNRESOLVED' if flags else 'NO_REPLACEMENT_CHAR_OBSERVED',
 original_strict_text_gate_overridden=False,quality='NOT_RUN',protocol='SourceFind forum reference 2026-09-22, retrieved 2026-10-17; 2 warmup, 3 formal, max(8,2C)',
 measured_rounds=3,requests=3*count,mean_output_tps=statistics.mean(vs),output_tps_rounds=vs,spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100,
 text_flags=flags,raw_sha256=manifest,config=cfg,official_final_acceptance=False)
for metric in ['ttft_s','tpot_ms','e2e_s']:
 for stat in ['mean','p50','p95','p99']:report[f'{metric}_{stat}_mean_of_rounds']=statistics.mean(s[metric][stat] for s in rounds)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
