#!/usr/bin/env python3
"""Read one completed run; keep per-round SLA decisions and report statistics distinct."""
import argparse
import json
from pathlib import Path

def read(path):
    return json.loads(path.read_text())

def summarize(directory):
    random_report = directory/'reference_performance_review.json'
    if random_report.exists():
        review = read(random_report)
        config = read(directory/'config.json')
        input_len = config['input_len']
        assert input_len in (4096, 65536), input_len
        limits = (10, 80) if input_len == 4096 else (30, 100)
        rounds = []
        for f in sorted(directory.glob('round-*/summary.json')):
            s = read(f)
            ttft, tpot = s['ttft_s']['p95'], s['tpot_ms']['p95']
            rounds.append(dict(round=f.parent.name, ttft_p95_s=ttft, tpot_p95_ms=tpot,
                sla_pass=ttft < limits[0] and tpot < limits[1]))
        assert len(rounds) == 3, 'A completed reference run needs three formal rounds'
        return dict(kind='fixed_random', directory=str(directory),
            performance_status=review['performance_status'], measurement_status=review['status'],
            mean_output_tps=review['mean_output_tps'],
            ttft_mean_s=review['ttft_s_mean_mean_of_rounds'],
            tpot_mean_ms=review['tpot_ms_mean_mean_of_rounds'],
            ttft_p95_s_mean_of_rounds=review['ttft_s_p95_mean_of_rounds'],
            tpot_p95_ms_mean_of_rounds=review['tpot_ms_p95_mean_of_rounds'],
            text_flagged_records=len(review['text_flags']),
            selected_sla=dict(ttft_p95_s_strict_lt=limits[0], tpot_p95_ms_strict_lt=limits[1],
                every_formal_round_pass=review['performance_status']=='PASS' and all(r['sla_pass'] for r in rounds),
                rounds=rounds, scope='One measured concurrency point; not a maximum-concurrency or official admission claim'),
            accuracy='NOT_MEASURED_BY_PERFORMANCE_RUN')
    sharegpt = directory/'validated_summary.json'
    if sharegpt.exists() and list(directory.glob('round-*.jsonl')):
        s = read(sharegpt)
        return dict(kind='sharegpt', directory=str(directory), validation=s,
            units='TPS: tokens/s; TTFT/TPOT/ITL fields ending in _ms: milliseconds',
            scope='This benchmark exposes TPOT median/P99; ITL P95 is not TPOT P95')
    gsm8k = directory/'summary.json'
    if gsm8k.exists() and read(gsm8k).get('benchmark') == 'GSM8K':
        return dict(kind='gsm8k', directory=str(directory), summary=read(gsm8k),
            independent_review=read(directory/'independent-review.json') if (directory/'independent-review.json').exists() else None)
    raise ValueError('No completed reference-performance, ShareGPT or GSM8K summary found; inspect the original log and failed run')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, help='One benchmark scene directory, or the GSM8K output directory')
    a = p.parse_args()
    print(json.dumps(summarize(a.directory), ensure_ascii=False, indent=2, allow_nan=False))
