#!/usr/bin/env python3
"""Provisional strict per-round P99 SLA check. Never asserts an official maximum."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('directories',type=Path,nargs='+')
a=p.parse_args(); rows=[]
for d in a.directories:
    cfg=json.loads((d/'config.json').read_text())
    if cfg['input_len'] not in [4096,65536] or cfg['output_len']!=1024:raise SystemExit('Unsupported workload')
    limits=(10,80) if cfg['input_len']==4096 else (30,100)
    files=sorted(d.glob('round-*/summary.json'))
    if len(files)!=3:raise SystemExit(f'Need three rounds: {d}')
    rr=[]
    for f in files:
        s=json.loads(f.read_text()); raw=[json.loads(line) for line in (f.parent/'requests.jsonl').read_text().splitlines()]
        valid=(s['status']=='PASS' and len(raw)==cfg['requests_per_round'] and
               s['successful']==len(raw) and all(not r['error'] and r['done'] and r['prompt_tokens']==cfg['input_len'] and r['completion_tokens']==1024 for r in raw))
        rr.append(dict(round=f.parent.name,valid=valid,ttft_p99_s=s['ttft_s']['p99'],tpot_p99_ms=s['tpot_ms']['p99'],
                       provisional_pass=valid and s['ttft_s']['p99']<limits[0] and s['tpot_ms']['p99']<limits[1]))
    rows.append(dict(input_len=cfg['input_len'],concurrency=cfg['concurrency'],rounds=rr,
                     provisional_all_rounds_pass=all(x['provisional_pass'] for x in rr)))
print(json.dumps(dict(status='PROVISIONAL_P99_SLA_ONLY',official_max_concurrency=None,
                     reason='Official statistic, duration, request count and boundary protocol not specified; no exhaustive boundary proof',rows=rows),indent=2))
