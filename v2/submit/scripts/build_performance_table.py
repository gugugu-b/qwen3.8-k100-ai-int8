#!/usr/bin/env python3
"""Build or verify one deduplicated table of every packaged v2 performance run.

CPU only. Original measurements, validators, and failed verdicts are read only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / 'result/summary/performance-all-concurrency.json'
DOC_PATH = ROOT / 'docs/性能总表.md'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, label):
    require(math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), label)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def json_compatible(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_compatible(v) for v in value]
    return value


def collect(scope=None):
    inventory = {}

    def read(path):
        inventory[path.relative_to(ROOT).as_posix()] = digest(path)
        return json.loads(path.read_text())

    if scope is None:
        scope = {'random': [], 'sharegpt': []}
        for first in sorted(ROOT.rglob('summary.json')):
            if first.parent.name != 'round-01':
                continue
            value = json.loads(first.read_text())
            if isinstance(value, dict) and {'input_len', 'output_throughput_tps'} <= value.keys():
                scope['random'].append(first.parent.parent.relative_to(ROOT).as_posix())
        for first in sorted(ROOT.rglob('round-01.jsonl')):
            value = json.loads(first.read_text())
            if isinstance(value, dict) and value.get('dataset_name') == 'sharegpt':
                scope['sharegpt'].append(first.parent.relative_to(ROOT).as_posix())
    for directories in scope.values():
        for relative in directories:
            path = Path(relative)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe input directory')
    groups = {}
    for relative in scope['random']:
        first = ROOT / relative / 'round-01/summary.json'
        value = read(first)
        if not isinstance(value, dict) or not {'input_len', 'output_throughput_tps'} <= value.keys():
            continue
        directory = first.parent.parent
        files = sorted(directory.glob('round-*/summary.json'))
        key = ('random', *(digest(p) for p in files))
        groups.setdefault(key, []).append(directory)
    for relative in scope['sharegpt']:
        first = ROOT / relative / 'round-01.jsonl'
        value = read(first)
        if not isinstance(value, dict) or value.get('dataset_name') != 'sharegpt':
            continue
        directory = first.parent
        files = sorted(directory.glob('round-*.jsonl'))
        key = ('sharegpt', *(digest(p) for p in files))
        groups.setdefault(key, []).append(directory)

    final = read(ROOT / 'logs/v2_benchmark_raw/final/manifest.json')
    final_meta = {str(ROOT / 'logs/v2_benchmark_raw/final' / s['directory']): s
                  for s in final['scenes']}
    high = read(ROOT / 'logs/v2_benchmark_raw/server64-128-20261017/summary.json')
    rows, excluded = [], []

    def rank(directory):
        rel = directory.relative_to(ROOT).as_posix()
        raw = bool(list(directory.glob('round-*/requests.jsonl'))
                   or list(directory.glob('round-*.jsonl')))
        return (not rel.startswith('logs/v2_benchmark_raw/final/'),
                not (directory / 'config.json').is_file(), not raw, len(rel), rel)

    for key, aliases in sorted(groups.items()):
        directory = sorted(aliases, key=rank)[0]
        rel = directory.relative_to(ROOT).as_posix()
        identifier = hashlib.sha256('|'.join(key).encode()).hexdigest()
        if re.search(r'(?:/v1/|/v1-return/|-v1/|-v1-return/)', '/' + rel + '/'):
            excluded.append({'reason': 'V1_OR_V1_RETURN_COMPARATOR', 'directory': rel,
                             'aliases': sorted(p.relative_to(ROOT).as_posix() for p in aliases),
                             'fingerprint': identifier})
            continue
        row = {'id': identifier, 'kind': key[0], 'directory': rel,
               'aliases': sorted(p.relative_to(ROOT).as_posix() for p in aliases),
               'tp': 2, 'observed_running_peak': None,
               'observed_running_peak_scope': 'NOT_MEASURED',
               'official_admission': 'NOT_ESTABLISHED'}
        summaries, flags = [], {'formal': 0, 'warmup': 0}
        if key[0] == 'random':
            config = read(directory / 'config.json')
            row.update(scene='4K/1K' if config['input_len'] == 4096 else '64K/1K',
                       concurrency=config['concurrency'], config=config)
            for p in sorted(directory.glob('round-*/summary.json')) + sorted(directory.glob('warmup-*/summary.json')):
                summary = read(p)
                raw = p.parent / 'requests.jsonl'
                require(raw.is_file(), 'Missing requests: ' + str(raw))
                inventory[raw.relative_to(ROOT).as_posix()] = digest(raw)
                requests = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
                require(len(requests) == summary['requested'] == summary['successful']
                        == summary['exact_length'], 'Request coverage: ' + str(p))
                require(len({r['request_index'] for r in requests}) == len(requests), 'Duplicate request index')
                require(all(r['completion_tokens'] == 1024 and not r['error'] for r in requests),
                        'Request error or incomplete output: ' + str(raw))
                close(sum(r['completion_tokens'] for r in requests) / summary['wall_time_s'],
                      summary['output_throughput_tps'], 'TPS raw arithmetic')
                for metric in ('ttft_s', 'tpot_ms', 'e2e_s'):
                    values = [r[metric] for r in requests]
                    close(statistics.mean(values), summary[metric]['mean'], metric + ' mean')
                    for q in (95, 99):
                        close(percentile(values, q), summary[metric]['p' + str(q)], metric + ' percentile')
                phase = 'formal' if summary['phase'] == 'measure' else 'warmup'
                flags[phase] += sum(bool(r.get('replacement_chars', 0)) for r in requests)
                if phase == 'formal':
                    summaries.append(summary)
            api = next((p for p in (directory / 'server-info.json', directory.parent / 'server-info.json')
                        if p.is_file()), None)
            require(api is not None, 'Missing service API: ' + rel)
            info = read(api)
            row['api_source'] = api.relative_to(ROOT).as_posix()
            limits = (10, 80) if row['scene'] == '4K/1K' else (30, 100)
            row['sla_rounds'] = [{'round': s['round'], 'ttft_p95_s': s['ttft_s']['p95'],
                                  'tpot_p95_ms': s['tpot_ms']['p95'],
                                  'passed': s['ttft_s']['p95'] < limits[0]
                                            and s['tpot_ms']['p95'] < limits[1]} for s in summaries]
            row['sla_passed_rounds'] = sum(r['passed'] for r in row['sla_rounds'])
            row['sla_limits'] = {'ttft_p95_s_lt': limits[0], 'tpot_p95_ms_lt': limits[1]}
            row['latency'] = {metric: {stat: statistics.mean(s[metric][stat] for s in summaries)
                                      for stat in ('mean', 'p95', 'p99')}
                              for metric in ('ttft_s', 'tpot_ms', 'e2e_s')}
            row['output_tps_rounds'] = [s['output_throughput_tps'] for s in summaries]
            row['formal_requests_per_round'] = [s['requested'] for s in summaries]
        else:
            files = sorted(directory.glob('round-*.jsonl'))
            summaries = [read(p) for p in files]
            first = summaries[0]
            info = first['server_info']
            row.update(scene='ShareGPT', concurrency=first['max_concurrency'],
                       api_source=(files[0].relative_to(ROOT).as_posix() + '#/server_info'),
                       sla_passed_rounds=None, sla_rounds=[], sla_limits=None)
            for s in summaries:
                require(s['completed'] == len(s['errors']) == 1000 and not any(s['errors']),
                        'ShareGPT request errors: ' + rel)
                require(s['input_lens'] == first['input_lens'] and s['max_concurrency'] == row['concurrency'],
                        'ShareGPT request set differs')
                close(sum(s['output_lens']) / s['duration'], s['output_throughput'], 'ShareGPT TPS')
            flags['formal'] = sum('\ufffd' in text for s in summaries for text in s['generated_texts'])
            for p in sorted(directory.glob('full-warmup-*.jsonl')):
                warm = read(p)
                flags['warmup'] += sum('\ufffd' in text for text in warm['generated_texts'])
            row['latency'] = {
                'ttft_s': {'mean': statistics.mean(s['mean_ttft_ms'] for s in summaries) / 1000,
                           'p95': None, 'p99': statistics.mean(s['p99_ttft_ms'] for s in summaries) / 1000},
                'tpot_ms': {'mean': statistics.mean(s['mean_tpot_ms'] for s in summaries),
                            'p95': None, 'p99': statistics.mean(s['p99_tpot_ms'] for s in summaries)},
                'e2e_s': {'mean': statistics.mean(s['mean_e2e_latency_ms'] for s in summaries) / 1000,
                          'p95': None, 'p99': statistics.mean(s['p99_e2e_latency_ms'] for s in summaries) / 1000}}
            row['output_tps_rounds'] = [s['output_throughput'] for s in summaries]
            row['formal_requests_per_round'] = [s['completed'] for s in summaries]
        args = info.get('server_args', info)
        require(args['tp_size'] == 2, 'Only TP2 results belong in this table')
        row.update(server_limit=args['max_running_requests'], graph_limit=args['cuda_graph_max_bs'],
                   formal_rounds=len(summaries), flags_formal=flags['formal'], flags_warmup=flags['warmup'],
                   mean_output_tps=statistics.mean(row['output_tps_rounds']) if len(summaries) == 3 else None)
        require(len(summaries) == 3, 'Incomplete run requires explicit handling: ' + rel)
        if str(directory) in final_meta:
            meta = final_meta[str(directory)]
            row['physical_pair'] = meta['physical_pair']
            if meta['role'] in ('primary_random', 'diagnostic_high_client_concurrency'):
                row.update(batch='参考矩阵C', protocol='forum-reference', order=3)
            elif meta['role'] == 'sla':
                row.update(batch='SLA复测D', protocol='forum-reference', order=4)
            elif meta['role'] == 'clean_runtime':
                row.update(batch='干净容器D', protocol='forum-reference', order=5)
            elif meta['kind'] == 'sharegpt':
                row.update(batch='ShareGPT复核D' if row['concurrency'] == 64 else 'ShareGPT原测',
                           protocol='sglang-bench-serving', order=4 if row['concurrency'] == 64 else 1)
            else:
                raise ValueError('Unclassified final scene: ' + rel)
        elif 'community-v2-20261016-a-' in rel:
            pair = re.search(r'-p([0246])([1357])-', rel)
            require(pair is not None, 'Missing pair')
            row.update(batch='初测A（旧口径）', protocol='internal-random', order=2,
                       physical_pair=pair[1] + ',' + pair[2])
        elif 'high-concurrency-20261017-a-' in rel:
            worker = next(w for w in high['workers'] if w['client_concurrency'] == row['concurrency'])
            require(len(worker['running_series']) == 1 and worker['formal_windows_used'], 'Ambiguous metrics')
            row.update(batch='容量复测', protocol='forum-reference', order=6, physical_pair=worker['pair'],
                       observed_running_peak=worker['running_series'][0]['peak_value'],
                       observed_running_peak_scope=worker['peak_scope'])
        elif '4k-delivery/' in rel:
            row.update(batch='交付复测10-10', protocol='internal-random', order=0, physical_pair='6,7')
        elif '64k-validation/' in rel:
            row.update(batch='候选复测10-16', protocol='internal-random', order=1, physical_pair='4,5')
        elif 'sharegpt-c4-c16-c64/' in rel:
            row.update(batch='ShareGPT原测', protocol='sglang-bench-serving', order=1, physical_pair='6,7')
        elif 'sharegpt-c64-server64-failed/' in rel:
            row.update(batch='服务64尝试（失败）', protocol='sglang-bench-serving', order=2, physical_pair='0,1',
                       original_failed_verdict='FAILED_TEXT_INTEGRITY_CHECK')
        else:
            raise ValueError('Unclassified candidate run: ' + rel)
        validation = next((p for p in (directory / 'reference_performance_review.json',
                                       directory / 'validated_summary.json') if p.is_file()), None)
        row['original_validation_source'] = validation.relative_to(ROOT).as_posix() if validation else None
        row['original_validation_status'] = read(validation).get('status') if validation else 'NO_PASS_REPORT'
        if flags['formal'] or flags['warmup']:
            row['text_status'] = '原文本门禁FAIL' if 'original_failed_verdict' in row else '文本标记未解决'
        else:
            row['text_status'] = '无U+FFFD标记'
        row['round_metrics'] = summaries if key[0] == 'random' else [
            {k: s[k] for k in s if k not in ('generated_texts', 'input_lens', 'output_lens', 'ttfts',
                                            'itls', 'errors', 'server_info', 'total_user_input_texts',
                                            'total_user_output_texts') and not isinstance(s[k], list)}
            for s in summaries]
        rows.append(row)
    rows.sort(key=lambda r: (('4K/1K', '64K/1K', 'ShareGPT').index(r['scene']),
                             r['concurrency'], r['order'], r['server_limit'], r['id']))
    for index, row in enumerate(rows, 1):
        row['row'] = index
    return json_compatible({'status': 'ALL_PACKAGED_V2_RUNS_RECALCULATED', 'rows': rows,
            'input_directories': scope,
            'row_count': len(rows), 'formal_rounds': sum(r['formal_rounds'] for r in rows),
            'concurrency_coverage': {s: sorted({r['concurrency'] for r in rows if r['scene'] == s})
                                     for s in ('4K/1K', '64K/1K', 'ShareGPT')},
            'raw_experiments_before_variant_filter': len(groups), 'excluded_v1_comparators': excluded,
            'deduplication': 'Exact SHA256 of all formal round summaries / ShareGPT raw JSONs; aliases retained',
            'aggregation': 'Arithmetic mean of three formal rounds; warmup excluded; percentiles averaged per round',
            'source_nonfinite_metadata': 'Original Infinity/NaN metadata is represented by strings; measurement columns must be finite',
            'gpu_used': False, 'measurements_modified': False, 'source_sha256': dict(sorted(inventory.items()))})


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def render_table(data, destination):
    lines = ['| 场景 | C | 批次／卡对 | 服务／Graph／实测峰值 | 正式轮×请求 | TPS R1／R2／R3 | **均值TPS** | TTFT mean／P95／P99（s） | TPOT mean／P95／P99（ms） | SLA通过轮 | U+FFFD正式／预热；文本状态 | 原始数据 |',
             '|---|---:|---|---|---|---|---:|---|---|---|---|---|']
    for row in data['rows']:
        peak = row['observed_running_peak']
        limits = f"{row['server_limit']}／{row['graph_limit']}／" + ('—' if peak is None else str(int(peak)))
        count = row['formal_requests_per_round']
        requests = f"{row['formal_rounds']}×{count[0]}" if len(set(count)) == 1 else '/'.join(map(str, count))
        sla = '—' if row['sla_passed_rounds'] is None else f"{row['sla_passed_rounds']}/{row['formal_rounds']}"
        link = os.path.relpath(ROOT / row['directory'], Path(destination).parent).replace(' ', '%20')
        batch = row['batch']
        if row['server_limit'] < row['concurrency']:
            batch += '（排队）'
        if peak is not None and peak < row['concurrency']:
            batch += '（未达' + str(row['concurrency']) + '）'
        vals = [row['scene'], str(row['concurrency']), batch + '；' + row['physical_pair'], limits,
                requests, '／'.join(map(fmt, row['output_tps_rounds'])), '**' + fmt(row['mean_output_tps']) + '**',
                '／'.join(fmt(row['latency']['ttft_s'][s]) for s in ('mean', 'p95', 'p99')),
                '／'.join(fmt(row['latency']['tpot_ms'][s]) for s in ('mean', 'p95', 'p99')), sla,
                f"{row['flags_formal']}/{row['flags_warmup']}；{row['text_status']}", '[原件](' + link + '/)']
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def render_environment(destination):
    source = ROOT / 'docs/实测环境声明.md'
    content = source.read_text().split('\n\n', 1)[1].strip()
    def rebase(match):
        target = match.group(1)
        if '://' in target or target.startswith('#'):
            return match.group(0)
        path, mark, fragment = target.partition('#')
        relative = os.path.relpath((source.parent / path).resolve(), destination.parent)
        return '](' + relative + (mark + fragment if mark else '') + ')'
    return re.sub(r'\]\(([^)]+)\)', rebase, content)


def render_document(data):
    return f'''# v2性能总表：全部已测并发与复测批次

{render_environment(DOC_PATH)}

本表覆盖包内 **{data['row_count']}组v2候选测量、{data['formal_rounds']}个正式轮**。扫描所有逐轮原件，按正式轮SHA去重；同一次测量的归档副本不重复计行，不同批次、协议、服务上限与复测全部保留。另有{len(data['excluded_v1_comparators'])}组v1／v1-return对照不冒充v2成绩，清单与全部来源见 [机器可读总表](../result/summary/performance-all-concurrency.json)。

4K覆盖C{','.join(map(str, data['concurrency_coverage']['4K/1K']))}；64K覆盖C{','.join(map(str, data['concurrency_coverage']['64K/1K']))}；ShareGPT覆盖C{','.join(map(str, data['concurrency_coverage']['ShareGPT']))}。各场景按客户端C升序排列，同一C的不同批次分别一行。

- TPS单位tokens/s，均值为三正式轮TPS的算术均值，排除预热；延迟为三轮同名统计的均值，P95／P99均值不是合并分位数。正式请求写作“轮数×每轮请求数”。
- “服务／Graph／实测峰值”前两项取服务API，第三项只有容量复测有正式时间窗内5秒采样证据，未采样填“—”。C128新配置观测峰值76；旧服务32的C64／128仍是排队诊断。
- SLA仅对随机负载按逐轮P95计算：4K TTFT＜10秒且TPOT＜80ms；64K＜30秒且＜100ms。“3/3”表示该点三轮通过，不能证明全局最大或模型精度。ShareGPT原件没有同口径TTFT／TPOT P95，填“—”，不拿ITL P95替代。
- “初测A”是内部旧脚本，正式请求每轮max(16,2C)；“参考矩阵C／SLA复测D／干净容器D／容量复测”是论坛reference脚本，每轮max(8,2C)。旧交付／候选复测单独保留；不同批次与同机并行条件不能合并成单项因果收益。
- U+FFFD列统计含标记的响应条数，正式与预热分列。ShareGPT服务64尝试已完整测完三轮，但原文本门禁FAIL，473.6951 TPS只作为失败测量保留，原失败判定不变。原始文本、请求完整性与最终赛事准入分别判断。

{render_table(data, DOC_PATH)}

复核命令（只读CPU，不发模型请求，不改原件）：

```bash
python3 -I scripts/build_performance_table.py --check
```

完整配置、每轮统计、原验证状态、重复副本路径、来源SHA与排除的v1对照均在JSON中；原始服务日志、早期失败和三轮原件继续保留。`--check`只读核验本版JSON列明的源目录，服务器后来新增的动态运行不改变本版快照；有意更新总表时再用`--write`完整发现源目录并重新封存。
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    a = p.parse_args()
    saved = json.loads(JSON_PATH.read_text()) if a.check else None
    data = collect(saved['input_directories'] if saved is not None else None)
    document = render_document(data)
    if a.write:
        JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        DOC_PATH.write_text(document)
    else:
        require(json.loads(JSON_PATH.read_text()) == data, 'Saved JSON differs from raw measurements')
        require(DOC_PATH.read_text() == document, 'Saved table differs from raw measurements')
        start, end = '<!-- V2_ALL_PERFORMANCE_TABLE:START -->', '<!-- V2_ALL_PERFORMANCE_TABLE:END -->'
        for path in (ROOT / 'README.md', ROOT / 'docs/性能优化报告.md', ROOT.parents[1] / 'README.md'):
            if path.is_file() and start in path.read_text():
                value = path.read_text().split(start, 1)[1].split(end, 1)[0]
                require(value == '\n' + render_table(data, path) + '\n', 'Embedded table differs: ' + str(path))
            if path.is_file() and '<!-- V2_MEASURED_ENV:START -->' in path.read_text():
                first_block = path.read_text().split('\n\n', 1)[1]
                require(first_block.startswith(render_environment(path)), 'Leading environment declaration differs: ' + str(path))
    print(json.dumps({'status': 'ALL_PERFORMANCE_TABLE_PASS', 'rows': data['row_count'],
                      'formal_rounds': data['formal_rounds'], 'coverage': data['concurrency_coverage'],
                      'excluded_v1_comparators': len(data['excluded_v1_comparators']),
                      'gpu_used': False, 'raw_files_modified': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
