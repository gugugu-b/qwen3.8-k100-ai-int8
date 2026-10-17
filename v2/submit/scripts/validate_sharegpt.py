#!/usr/bin/env python3
"""Validate full ShareGPT runs and keep TTFT/TPOT/ITL units distinct."""
import argparse
import json
import math
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--rounds',type=int,default=3)
p.add_argument('--full-warmup-rounds',type=int,default=0)
a=p.parse_args()
files=sorted(a.directory.glob('round-*.jsonl'))
assert len(files)==a.rounds, (len(files),a.rounds)
reports=[]
for f in files:
    lines=[s for s in f.read_text().splitlines() if s.strip()]
    assert len(lines)==1, f
    r=json.loads(lines[0])
    assert r['completed']==1000 and r['dataset_name']=='sharegpt', f
    assert len(r['errors'])==1000 and not any(r['errors']), f
    for key in ['input_lens','output_lens','generated_texts','ttfts','itls']:
        assert len(r[key])==1000, (f,key)
    assert all(v>0 for v in r['input_lens']) and all(v>0 for v in r['output_lens']), f
    assert not any('\ufffd' in t for t in r['generated_texts']), f
    assert math.isfinite(r['output_throughput']) and r['output_throughput']>0, f
    assert math.isfinite(r['duration']) and r['duration']>0, f
    reports.append(r)
assert all(r['input_lens']==reports[0]['input_lens'] for r in reports), 'request sets differ'
assert all(r['max_concurrency']==reports[0]['max_concurrency'] for r in reports)
warmfiles=sorted(a.directory.glob('full-warmup-*.jsonl'))
assert len(warmfiles)==a.full_warmup_rounds,(len(warmfiles),a.full_warmup_rounds)
for f in warmfiles:
    lines=[s for s in f.read_text().splitlines() if s.strip()]
    assert len(lines)==1,f
    w=json.loads(lines[0])
    assert w['completed']==1000 and len(w['errors'])==1000 and not any(w['errors']),f
    assert w['input_lens']==reports[0]['input_lens'],'Full warmup request set differs'
    assert w['max_concurrency']==reports[0]['max_concurrency']
vs=[r['output_throughput'] for r in reports]
summary=dict(status='PASS',rounds=len(files),requests=1000*len(files),errors=0,
    full_workload_warmup_rounds=len(warmfiles),short_warmup_requests_per_run=64,
    client_concurrency=reports[0]['max_concurrency'],output_tps_rounds=vs,
    mean_output_tps=statistics.mean(vs),spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100,
    mean_duration_s=statistics.mean(r['duration'] for r in reports))
for key in ['median_ttft_ms','p99_ttft_ms','median_tpot_ms','p99_tpot_ms',
            'median_itl_ms','p95_itl_ms','p99_itl_ms','p99_e2e_latency_ms']:
    values=[r[key] for r in reports]
    assert all(math.isfinite(v) and v>=0 for v in values), key
    summary[key+'_mean_of_rounds']=statistics.mean(values)
    summary[key+'_range']=[min(values),max(values)]
(a.directory/'validated_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
