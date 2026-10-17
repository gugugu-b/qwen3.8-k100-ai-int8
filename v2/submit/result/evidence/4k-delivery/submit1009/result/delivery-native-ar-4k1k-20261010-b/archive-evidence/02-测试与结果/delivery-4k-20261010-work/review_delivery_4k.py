#!/usr/bin/env python3
"""Independent CPU review of immutable three-variant 4K/1K evidence."""
import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
RUN = 'delivery-native-ar-4k1k-20261010-b'
root = a.root.resolve()
control = root / 'result' / RUN
a.out.mkdir(parents=True, exist_ok=False)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

manifest = json.loads((control / 'complete-sha256.json').read_text())
for rel, digest in manifest.items():
    path = root / rel
    assert path.is_file() and path.resolve().is_relative_to(root), rel
    assert sha(path) == digest, rel

def percentile(vs, q):
    vs = sorted(vs)
    pos = (len(vs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return vs[lo] if lo == hi else vs[lo] + (vs[hi] - vs[lo]) * (pos - lo)

summaries, info, cfg, all_rounds = {}, {}, {}, {}
for kind in ('v1', 'native-bf16-ar', 'v1-return'):
    folder = root / 'result' / (RUN + '-' + kind)
    scene = folder / '4k'
    cfg[kind] = json.loads((scene / 'config.json').read_text())
    for key, value in dict(input_len=4096, output_len=1024, concurrency=8, seed=20260912,
                           warmup_rounds=2, measure_rounds=3, requests_per_round=16).items():
        assert cfg[kind][key] == value, (kind, key)
    for phase, count, n in [('warmup', 2, 8), ('round', 3, 16)]:
        files = sorted(scene.glob(phase + '-*/requests.jsonl'))
        assert len(files) == count, (kind, phase)
        for f in files:
            rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            assert len(rows) == n and sorted(r['request_index'] for r in rows) == list(range(n)), f
            for r in rows:
                assert r['done'] and not r['error'] and r['status_code'] == 200, f
                assert r['prompt_tokens'] == 4096 and r['completion_tokens'] == 1024, f
                assert r['replacement_chars'] == 0, f
                assert all(math.isfinite(r[k]) and r[k] >= 0 for k in ('e2e_s', 'ttft_s', 'tpot_ms')), f
                expected = 1000 * (r['e2e_s'] - r['ttft_s']) / 1023
                assert math.isclose(expected, r['tpot_ms'], rel_tol=1e-12), f
            s = json.loads((f.parent / 'summary.json').read_text())
            assert s['status'] == 'PASS' and s['successful'] == s['exact_length'] == n, f
            assert s['input_len'] == 4096 and s['output_len'] == 1024 and s['concurrency'] == 8, f
            assert math.isclose(s['output_throughput_tps'], n * 1024 / s['wall_time_s'], rel_tol=1e-12), f
            for metric in ('e2e_s', 'ttft_s', 'tpot_ms'):
                values = [r[metric] for r in rows]
                assert s[metric]['count'] == n, f
                assert math.isclose(s[metric]['mean'], sum(values) / n, rel_tol=1e-12), f
                for q in (50, 90, 95, 99):
                    assert math.isclose(s[metric]['p' + str(q)], percentile(values, q), rel_tol=1e-12), f
    copy = a.out / kind
    copy.mkdir()
    for f in scene.glob('round-*'):
        shutil.copytree(f, copy / f.name)
    subprocess.run(['python3', str(root / 'scripts/validate_results.py'), str(copy), '--rounds', '3', '--input-len', '4096'],
                   check=True, stdout=subprocess.DEVNULL)
    assert (copy / 'validated_summary.json').read_bytes() == (scene / 'validated_summary.json').read_bytes(), kind
    summaries[kind] = json.loads((copy / 'validated_summary.json').read_text())
    info[kind] = json.loads((folder / 'server-info.json').read_text())
    all_rounds[kind] = [json.loads((f / 'summary.json').read_text()) for f in sorted(scene.glob('round-*'))]
    backend = 'native' if kind == 'native-bf16-ar' else 'off'
    assert info[kind]['custom_all_reduce_backend'] == backend
    assert info[kind]['disable_custom_all_reduce'] is (kind != 'native-bf16-ar')

same_fields = ['tp_size', 'pp_size', 'attention_backend', 'mm_attention_backend', 'page_size',
               'context_length', 'disable_radix_cache', 'mem_fraction_static', 'max_running_requests',
               'max_prefill_tokens', 'chunked_prefill_size', 'mamba_scheduler_strategy', 'cuda_graph_max_bs',
               'disable_cuda_graph', 'speculative_algorithm', 'speculative_num_steps', 'speculative_eagle_topk',
               'speculative_num_draft_tokens']
for key in same_fields:
    assert key in info['v1'], key
    assert len({json.dumps(s[key], sort_keys=True) for s in info.values()}) == 1, key
assert info['v1']['tp_size'] == 2 and info['v1']['disable_cuda_graph'] is False
base = statistics.mean([summaries['v1']['mean_output_tps'], summaries['v1-return']['mean_output_tps']])
cand = summaries['native-bf16-ar']['mean_output_tps']
gain = (cand / base - 1) * 100
drift = (summaries['v1-return']['mean_output_tps'] / summaries['v1']['mean_output_tps'] - 1) * 100
comparison = json.loads((control / 'comparison.json').read_text())
assert comparison['summaries'] == summaries
assert comparison['combined_v1_tps'] == base and comparison['native_ar_gain_vs_v1_percent'] == gain
assert comparison['v1_return_drift_percent'] == drift and comparison['rounds_per_variant'] == 3
paired = [(summaries['native-bf16-ar']['output_tps_rounds'][i] /
           statistics.mean([summaries['v1']['output_tps_rounds'][i], summaries['v1-return']['output_tps_rounds'][i]]) - 1) * 100
          for i in range(3)]
noise = max(abs(drift), *(s['spread_percent'] for s in summaries.values()))
output_mismatches = {}
candidate_by_id = {(i // 16, r['request_index']): r['output_sha256'] for i,r in enumerate(
    [r for f in sorted((root / 'result' / (RUN + '-native-bf16-ar') / '4k').glob('round-*/requests.jsonl'))
     for r in sorted([json.loads(line) for line in f.read_text().splitlines() if line.strip()], key=lambda x:x['request_index'])])}
for kind in ('v1', 'v1-return'):
    mismatches = 0
    for i,f in enumerate(sorted((root / 'result' / (RUN + '-' + kind) / '4k').glob('round-*/requests.jsonl'))):
        for r in [json.loads(line) for line in f.read_text().splitlines() if line.strip()]:
            mismatches += r['output_sha256'] != candidate_by_id[(i, r['request_index'])]
    output_mismatches[kind] = mismatches
tails, tails_initial = {}, {}
for metric in ('ttft_s', 'tpot_ms', 'e2e_s'):
    for q in (95, 99):
        k = metric + '_p' + str(q)
        c = statistics.mean(s[metric]['p' + str(q)] for s in all_rounds['native-bf16-ar'])
        before = statistics.mean(s[metric]['p' + str(q)] for s in all_rounds['v1'])
        after = statistics.mean(s[metric]['p' + str(q)] for s in all_rounds['v1-return'])
        tails[k] = (c / statistics.mean([before, after]) - 1) * 100
        tails_initial[k] = (c / before - 1) * 100
report = dict(status='THREE_ROUND_4K1K_RAW_PERFORMANCE_REVIEW_PASS', run_id=RUN,
 raw_files=len(manifest), raw_manifest_sha256=sha(control / 'complete-sha256.json'),
 workload=dict(input_len=4096, output_len=1024, concurrency=8, seed=20260912, physical_pair='6,7', tp=2,
               warmup_rounds=2, warmup_requests_per_round=8, measured_rounds_per_variant=3, measured_requests_per_variant=48),
 measured_requests=144, warmup_requests=48, errors=0, summaries=summaries, combined_v1_tps=base,
 candidate_mean_tps=cand, gain_vs_combined_v1_percent=gain, paired_round_gains_percent=paired,
 v1_return_drift_percent=drift, observed_noise_percent=noise, gain_exceeds_observed_noise=gain > noise,
 noise_method='Conservative observed variation reference: max of group ranges and baseline drift; round inputs differ and this is not a statistical noise bound',
 output_sha256_mismatch_requests_vs_baselines=output_mismatches,
 output_comparison_scope='Only the 48 deterministic synthetic prompts per variant; not a GSM8K or semantic accuracy gate',
 tail_regressions_vs_combined_v1_percent=tails, tail_regressions_vs_initial_v1_percent=tails_initial,
 tail_limit_percent=5, measured_tail_gate_pass=max(tails.values()) <= 5 and max(tails_initial.values()) <= 5,
 required_gain_percent=20, measured_4k_20pct_pass=gain >= 20 and all(g >= 20 for g in paired),
 actual_model_graph_gate='REVIEW_REQUIRED', clean_room='NOT_RUN', gsm8k='NOT_RUN',
 formal_acceptance=False, frozen=False, other_scenes='64K and ShareGPT C32 not rerun in timed delivery',
 protocol='Three rounds per consecutive variant; v1 before and after bracket candidate. Same physical pair and workload.')
(a.out / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:report[k] for k in ('status','combined_v1_tps','candidate_mean_tps','gain_vs_combined_v1_percent',
 'v1_return_drift_percent','observed_noise_percent','measured_tail_gate_pass','measured_4k_20pct_pass')}))
