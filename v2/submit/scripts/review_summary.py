#!/usr/bin/env python3
"""CPU recheck source SHA, three-round means and historical deltas. Writes no evidence."""
import hashlib
import json
import math
import statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
x=json.loads((root/'result/summary/final-performance-report-20261016.json').read_text())
for rel,e in x['source_inventory']['files'].items():
    assert hashlib.sha256((root/'result/summary'/rel).read_bytes()).hexdigest()==e['sha256'],rel
for row in x['rows']:
    vs=row['output_tps_rounds']; assert len(vs)==3 and all(math.isfinite(v) and v>0 for v in vs)
    mean=statistics.mean(vs)
    assert math.isclose(mean,row['mean_output_tps'],abs_tol=1e-9)
    pct=(mean/row['historical_v1_report_output_tps']-1)*100
    assert math.isclose(pct,row['historical_output_tps_change_percent'],abs_tol=1e-9)
    # Recheck the actual raw summary for every selected round.
    for rr in row['per_round']:
        raw=json.loads((root/'result/summary'/rr['source']['input']).read_text())
        tps=raw['output_throughput_tps'] if 'output_throughput_tps' in raw else raw['output_throughput']
        assert math.isclose(tps,rr['output_tps'],abs_tol=1e-9)
print(json.dumps({'status':'SEVEN_SCENE_SUMMARY_RECHECK_PASS','scenes':len(x['rows']),
                  'scope':'source SHA, original round TPS, three-round means, historical-reference arithmetic only',
                  'accuracy':'NOT_RUN','official_scoring':'NOT_VERIFIED'},indent=2))
