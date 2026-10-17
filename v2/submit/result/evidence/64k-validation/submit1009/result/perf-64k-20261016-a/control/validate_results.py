#!/usr/bin/env python3
"""Fail closed on incomplete, stale, or invalid fixed-length benchmark results."""
import argparse
import json
import math
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--input-len', type=int, required=True)
p.add_argument('--requests', type=int, default=16)
a = p.parse_args()
files = sorted(a.directory.glob('round-*/requests.jsonl'))
assert len(files) == a.rounds, f'expected {a.rounds} rounds, got {len(files)}'
summaries = []
for f in files:
    rows = [json.loads(s) for s in f.read_text().splitlines() if s.strip()]
    assert len(rows) == a.requests, (f, len(rows))
    for r in rows:
        assert not r.get('error'), (f, r.get('error'))
        assert r['prompt_tokens'] == a.input_len and r['completion_tokens'] == 1024, f
        assert r.get('replacement_chars', 0) == 0, f
    s = json.loads((f.parent/'summary.json').read_text())
    assert s['status'] == 'PASS' and s['successful'] == a.requests, f
    assert math.isfinite(s['output_throughput_tps']) and s['output_throughput_tps'] > 0, f
    assert math.isfinite(s['wall_time_s']) and s['wall_time_s'] > 0, f
    summaries.append(s)
tps = [s['output_throughput_tps'] for s in summaries]
report = {'status': 'PASS', 'measured_rounds': len(files), 'requests': a.rounds*a.requests,
          'output_tps_rounds': tps, 'mean_output_tps': statistics.mean(tps),
          'spread_percent': (max(tps)-min(tps))/statistics.mean(tps)*100,
          'mean_wall_s': statistics.mean(s['wall_time_s'] for s in summaries)}
for metric in ['ttft_s', 'tpot_ms']:
    for quantile in ['p50','p95','p99']:
        vs = [s[metric][quantile] for s in summaries]
        report[f'{metric}_{quantile}_mean_of_rounds'] = statistics.mean(vs)
        report[f'{metric}_{quantile}_range'] = [min(vs), max(vs)]
(a.directory/'validated_summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
