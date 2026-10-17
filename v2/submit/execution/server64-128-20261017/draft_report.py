#!/usr/bin/env python3
"""Draft local Markdown and JSON from a completed independent CPU review.

Writes only three new files; existing output files are never overwritten. No SSH, GPU, subprocess,
model requests, or edits to the submitted README/report are performed.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
KV_ALLOCATION = re.compile(r'KV Cache is allocated\.\s*#tokens:\s*(\d+),\s*'
                           r'K size:\s*([\d.]+)\s*(\w+),\s*V size:\s*([\d.]+)\s*(\w+)', re.I)
ROUTING_LABEL = re.compile(r'(?:^|[{,\s])routing_key\s*=')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def file_record(path, role):
    path = Path(path)
    return {'role': role, 'path': str(path), 'available': path.is_file(),
            **({'bytes': path.stat().st_size, 'sha256': sha(path)} if path.is_file() else {})}


def link(path, label, destination):
    relative = os.path.relpath(str(path), str(destination))
    return '[' + label + '](' + relative.replace(' ', '%20') + ')'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def fmt(value, places=4):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'{value:.{places}f}'
    return '—'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows])


def exact_running_series(metrics):
    # Each original label series remains independent. A routing-key bucket is
    # never used as the service-wide request-count gauge, even if aliased.
    return [entry for entry in metrics.get('series', [])
            if entry.get('metric_name') == 'sglang:num_running_reqs'
            and not ROUTING_LABEL.search(entry.get('original_labels', ''))]


def select_metrics(worker):
    formal = worker.get('formal_window_metrics', {})
    evidence = worker.get('formal_window_evidence', {})
    windows = evidence.get('windows', [])
    all_windows = (evidence.get('status') == 'ALL_THREE_FORMAL_WINDOWS_RECORDED'
                   and len(windows) == 3 and {row.get('round') for row in windows} == {1, 2, 3}
                   and evidence.get('inward_margin_s') == 5)
    if (all_windows and formal.get('status') == 'RAW_SAMPLES_PARSED'
            and formal.get('http_success_samples', 0) > 0 and exact_running_series(formal)):
        return formal, '三正式轮窗口内部（前后各向内5秒）', True
    return worker.get('metrics', {}), '整个Benchmark采样期，包含预热', False


def cache_evidence(worker, review):
    records = []
    api_budgets = []
    api_path = worker.get('api_capacity', {}).get('path')
    if api_path and Path(api_path).is_file():
        api = json.loads(Path(api_path).read_text())
        def walk(value, pointer=''):
            if isinstance(value, dict):
                for key, child in value.items():
                    location = pointer + '/' + key
                    if key in ('token_capacity', 'max_total_num_tokens') and isinstance(child, int) and child > 0:
                        api_budgets.append({'json_pointer': location, 'tokens': child,
                                            'path': api_path, 'sha256': sha(Path(api_path))})
                    walk(child, location)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, pointer + '/' + str(index))
        walk(api)
    control = Path(review.get('control', ''))
    key = worker.get('pair', '').replace(',', '')
    runtime = control / ('runtime-p' + key)
    candidates = [runtime / 'server-final/server.log']
    collection = Path(review.get('collection_root', ''))
    run_id = review.get('run_id', '')
    candidates.append(collection / 'submit/logs' / f'{run_id}-p{key}-server/server.log')
    # These two paths normally contain identical copies; select one original
    # log rather than duplicating cache rows from both copies.
    source = next((path for path in candidates if path.is_file()), None)
    budgets = []
    if source:
        source_sha = sha(source)
        for index, line in enumerate(source.read_text(errors='replace').splitlines(), 1):
            allocation = KV_ALLOCATION.search(line)
            if allocation:
                tokens, k_size, k_unit, v_size, v_unit = allocation.groups()
                records.append({'path': str(source), 'sha256': source_sha, 'line_number': index,
                                'tokens': int(tokens), 'k_size': float(k_size), 'k_unit': k_unit,
                                'v_size': float(v_size), 'v_unit': v_unit, 'original_line': line})
            budget = re.search(r'\bmax_total_num_tokens\s*=\s*(\d+)', line)
            if budget:
                budgets.append({'tokens': int(budget.group(1)), 'path': str(source),
                                'sha256': source_sha, 'line_number': index, 'original_line': line})
    return {'scheduler_budget_records': budgets, 'api_budget_records': api_budgets,
            'kv_allocations': records, 'source_log': str(source) if source else None,
            'does_not_convert_token_budget_into_verified_request_concurrency': True}


def source_evidence(review, review_path):
    local = review_path.parent
    records = []
    executed = review.get('controller-source.py', {}).get('path')
    if executed:
        records.append(file_record(executed, '实际执行controller完整副本'))
    control = Path(review.get('control', ''))
    for key in ('45', '67'):
        for name in ('serve.sh', 'start_server.sh'):
            records.append(file_record(control / f'runtime-p{key}/scripts' / name,
                                       f'物理卡对{key}实际私有{name}'))
    records.append(file_record(control / 'preparation.json', '准备阶段辅助源码身份记录'))
    source_package = Path(review.get('source_base', str(WORKSPACE))) / 'v2/submit'
    for relative, role in (
            ('execution/official-alignment-20261017-c/controller.py', '固定官方协议辅助controller源码'),
            ('execution/official-alignment-20261017-c/probe_template.py', '固定数值Graph门禁模板源码'),
            ('scripts/benchmark_reference.sh', '固定随机协议命令入口'),
            ('scripts/benchmark.reference.py', '固定论坛benchmark完整源码')):
        records.append(file_record(source_package / relative, role))
    for path, role in ((local / 'source-executed/controller.py', '派发时执行源码副本'),
                       (local / 'source-executed/dispatch.json', '派发记录与执行SHA'),
                       (local / 'executed-vs-terminal-handler-update.diff', '实际执行版与后续终态处理修改的Diff'),
                       (Path(__file__).with_name('review_results.py'), '本次CPU独立复核源码'),
                       (Path(__file__).with_name('collect.py'), '原始采集源码'),
                       (Path(__file__), '本报告生成源码')):
        records.append(file_record(path, role))
    actual = next((row for row in records if row['role'] == '实际执行controller完整副本' and row['available']), None)
    dispatch_path = local / 'source-executed/dispatch.json'
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.is_file() else {}
    return {'files': records, 'actual_executed_controller_sha256': actual.get('sha256') if actual else None,
            'dispatch_source_sha256': dispatch.get('source_sha256'),
            'dispatch_matches_actual_executed_copy': (
                actual['sha256'] == dispatch.get('source_sha256') if actual and dispatch.get('source_sha256') else None),
            'source_diff_is_provenance_only_execution_identity_comes_from_collected_source': True}


def summarize_worker(worker, review):
    concurrency = worker.get('requested_client_concurrency')
    require(concurrency in (64, 128), 'Unexpected worker concurrency')
    metrics, scope, formal = select_metrics(worker)
    running = exact_running_series(metrics)
    api = worker.get('api_capacity', {})
    actual = api.get('actual', {}) if api.get('status') == 'ACTUAL_API_CAPACITY_VERIFIED' else {}
    complete = [scene for scene in worker.get('scenes', [])
                if scene.get('status') == 'PERFORMANCE_RAW_REVIEW_REPRODUCED'
                and scene.get('formal_rounds') == 3]
    scene = complete[0] if len(complete) == 1 else None
    old = worker.get('previous_server32', {})
    observed_at_C = any(entry.get('peak_value', -1) >= concurrency for entry in running)
    if not running:
        capacity = '未找到可用的精确num_running_reqs原标签系列，实际并发未证实'
    elif observed_at_C:
        capacity = ('所列原标签系列在' + ('正式窗口内' if formal else '含预热采样期')
                    + '观测到请求数达到配置C；不将采样峰值等同完整容量／准入通过')
    else:
        capacity = '所列原标签系列峰值未达到配置C，不能称实际并发C已验证；保留KV预算与排队证据'
    return {'pair': worker.get('pair'), 'client_concurrency': concurrency,
            'worker_status': worker.get('worker_status'), 'review_status': worker.get('review_status'),
            'api_server_limit': actual.get('max_running_requests'),
            'api_graph_limit': actual.get('cuda_graph_max_bs'),
            'api_evidence': {'status': api.get('status'), 'path': api.get('path'), 'sha256': api.get('sha256')},
            'running_metric_name': 'sglang:num_running_reqs', 'running_series': running,
            'peak_scope': scope, 'formal_windows_used': formal,
            'formal_window_evidence': worker.get('formal_window_evidence'),
            'metrics_evidence': {key: metrics.get(key) for key in
                ('path', 'sha256', 'raw_samples', 'http_success_samples', 'error_samples',
                 'parse_errors', 'scope', 'selection_windows')},
            'queue_series': [entry for entry in metrics.get('series', [])
                             if entry.get('category') == 'queue_gauge'
                             and not ROUTING_LABEL.search(entry.get('original_labels', ''))],
            'gauge_aggregation': 'NONE; every original label series is separate',
            'capacity_interpretation': capacity, 'cache_evidence': cache_evidence(worker, review),
            'formal_rounds_complete': bool(scene),
            'formal_requests_expected': 6 * concurrency, 'warmup_requests_expected': 2 * concurrency,
            'formal_requests_per_round': 2 * concurrency, 'warmup_requests_per_round': concurrency,
            'output_tps_rounds': scene.get('output_tps_rounds') if scene else None,
            'three_round_mean': scene.get('three_round_mean') if scene else None,
            'flags_formal_count': len(scene.get('flags_formal', [])) if scene else None,
            'flags_warmup_count': len(scene.get('flags_warmup', [])) if scene else None,
            'original_text_integrity_status': scene.get('original_text_integrity_status') if scene else None,
            'scene_directory': scene.get('directory') if scene else None,
            'original_review_sha256': scene.get('original_review_sha256') if scene else None,
            'resource_release_status': worker.get('resource_release_status'),
            'lock_released_reported': worker.get('lock_released_reported'),
            'gpu_after_snapshot_idle': worker.get('gpu_after_snapshot_idle'),
            'container_stopped': worker.get('container_stopped'),
            'failure': {'error': worker.get('error'), 'issues': worker.get('issues'),
                        'capacity_evidence': worker.get('capacity_failure_evidence')},
            # Deliberately do not propagate the removed old S32 throughput numbers.
            'previous_S32_source_and_conditions_only': {key: old.get(key) for key in
                ('path', 'sha256', 'physical_pair', 'client_concurrency',
                 'server_max_running_requests', 'cuda_graph_max_bs')}}


def render(summary, destination):
    rows = summary['workers']
    capacity_rows = []
    series_rows = []
    performance_rows = []
    latency_rows = []
    resource_rows = []
    cache_rows = []
    for row in rows:
        c = row['client_concurrency']
        names = []
        for index, series in enumerate(row['running_series'], 1):
            identifier = f'R{c}-{index}'
            names.append(identifier + '=' + fmt(series['peak_value'], 0))
            series_rows.append([identifier, series['metric_name'], '`' + series.get('original_labels', '') + '`',
                                fmt(series['peak_value'], 0), series.get('peak_utc'), row['peak_scope'],
                                series.get('peak_sample')])
        capacity_rows.append([c, row['pair'], row['api_server_limit'] if row['api_server_limit'] is not None else '未核实',
                              row['api_graph_limit'] if row['api_graph_limit'] is not None else '未核实',
                              '；'.join(names) if names else '未取到精确指标', row['peak_scope'], row['review_status']])
        tps = row['output_tps_rounds'] or [None, None, None]
        means = row['three_round_mean'] or {}
        performance_rows.append([c, *[fmt(value) for value in tps], fmt(means.get('output_tps')),
                                 row['flags_formal_count'] if row['flags_formal_count'] is not None else '未完成',
                                 row['flags_warmup_count'] if row['flags_warmup_count'] is not None else '未完成'])
        latency_rows.append([c, *[fmt(means.get(key)) for key in
            ('ttft_s_mean', 'ttft_s_p95', 'ttft_s_p99', 'tpot_ms_mean', 'tpot_ms_p95', 'tpot_ms_p99')]])
        resource_rows.append([c, row['worker_status'], row['resource_release_status'],
                              row['lock_released_reported'], row['gpu_after_snapshot_idle']])
        cache = row['cache_evidence']
        for record in cache['scheduler_budget_records']:
            cache_rows.append([c, record['tokens'], 'scheduler max_total_num_tokens',
                               link(record['path'], '原日志:' + str(record['line_number']), destination)])
        for record in cache['kv_allocations']:
            cache_rows.append([c, record['tokens'],
                f"K={record['k_size']} {record['k_unit']}，V={record['v_size']} {record['v_unit']}",
                link(record['path'], '原日志:' + str(record['line_number']), destination)])
    capacity = table(['客户端C', '物理卡对', 'API服务上限', 'API Graph上限',
                      '实际num_running_reqs峰值（逐标签）', '采样范围', '独立复核状态'], capacity_rows)
    performance = table(['客户端C', '正式轮1 TPS', '正式轮2 TPS', '正式轮3 TPS', '三轮均值 TPS',
                         'U+FFFD正式响应数', 'U+FFFD预热响应数'], performance_rows)
    latency = table(['客户端C', 'TTFT mean s', 'TTFT P95 s', 'TTFT P99 s',
                     'TPOT mean ms', 'TPOT P95 ms', 'TPOT P99 ms'], latency_rows)
    metric_table = table(['系列', '精确指标名', '原标签', '峰值', '样本UTC（GET前）', '范围', '原样本行号'], series_rows)
    sources = '\n'.join('- ' + row['role'] + '：' + (link(row['path'], Path(row['path']).name, destination)
        + '，SHA256 `' + row['sha256'] + '`' if row['available'] else '未归档')
        for row in summary['source_trace']['files'])
    interpretations = '\n\n'.join(f"C{row['client_concurrency']}：{row['capacity_interpretation']}。" for row in rows)
    queues = []
    for row in rows:
        for series in row['queue_series']:
            queues.append([row['client_concurrency'], series['metric_name'],
                           '`' + series.get('original_labels', '') + '`', fmt(series.get('peak_value'), 0),
                           series.get('peak_utc'), row['peak_scope']])
    old_links = []
    for row in rows:
        old = row['previous_S32_source_and_conditions_only']
        path = old.get('path')
        old_links.append(f"- C{row['client_concurrency']}：" +
            (link(path, '旧S32原始来源', destination) if path else '旧来源未提供') +
            f"；原物理卡对{old.get('physical_pair')}，服务上限{old.get('server_max_running_requests')}／Graph{old.get('cuda_graph_max_bs')}。")
    failures = []
    for row in rows:
        if row['failure'].get('error') or row['failure'].get('issues') or not row['formal_rounds_complete']:
            failures.append(f"- C{row['client_concurrency']}：" + cell(json.dumps(row['failure'], ensure_ascii=False)))
    report = f'''# 4K/1K 服务64／128与实际运行并发复测

本报告由独立CPU复核结果生成，run `{summary['run_id']}`。复核状态：`{summary['status']}`；没有以配置值代替实际运行并发，没有将缺轮或失败数据补成成绩。结果来源：{link(summary['review_path'], '独立review JSON', destination)}。

## 配置与实际观测

{capacity}

{interpretations}

实际并发只引用精确指标 `sglang:num_running_reqs` 的各原标签系列，既不采用routing_key桶，也不对rank／标签求和。下表每行都是独立系列峰值，没有“合计峰值”。配置C128不能单独证明实际运行128个请求。

{metric_table}

若三正式轮绝对窗口全部有证据，优先采用每轮 `started_at + wall_time_s` 且前后各向内5秒的样本；UTC为HTTP GET前记录，配置超时4秒。否则表中明确使用含预热的整个Benchmark采样期。5秒离散采样的观察峰值不是连续监控上界，也不等于完整容量验证。

## 三正式轮性能与文本标记

{performance}

{latency}

TPS为输出tokens/s；只计算三次正式轮TPS的算术均值，不计预热，也不以总Token／总墙钟替换。延迟列为三轮对应mean／P95／P99的均值，分位数均值不是合并总体分位数。文本标记按正式与预热响应分别计数，性能原始数据复核通过不能代替文本、精度或赛事准入。

请求配置：4096输入、1024输出、ignore_eos、seed20260912。C64两轮预热合计128请求、三正式轮合计384请求（每轮128）；C128两轮预热合计256请求、三正式轮合计768请求（每轮256）。缺轮时上述只是计划请求数，表中不填写三轮成绩。

## KV预算、缓存与排队证据

{table(['客户端C', 'KV Token预算／分配Token数', '记录类型／K与V大小（保持原日志单位）', '来源'], cache_rows) if cache_rows else '未从已提供的原始日志提取到KV预算／K、V分配记录；不补写推测值。'}

{table(['客户端C', '排队Gauge', '原标签', '峰值', '样本UTC', '范围'], queues) if queues else '未取得可用排队Gauge。'}

KV预算及K／V大小保持每条原日志记录，不跨rank或主／draft缓存相加。调度可运行请求数同时受到Token预算与实际输入／输出长度等条件约束；API上限128不消除缓存预算限制。若实测系列峰值未到128，报告只能保留该观测及排队记录，不能宣称实际128并发容量通过。

## 实验条件与资源释放

两实例分别TP2：C64物理4,5，C128物理6,7，同一主机并行；共享主机／IO／带宽干扰条件保留。除服务max_running_requests与Graph上限匹配C外，沿用候选、固定镜像、context67584、显存0.85、MTP与prefill参数。正式采样增加了每5秒一次 `/metrics` 的轻量HTTP GET，未运行Profiler／trace。

{table(['客户端C', 'worker终态', '停止／资源证据', '报告锁已释放', 'GPU退出快照空闲'], resource_rows)}

锁释放为控制器记录，GPU为退出时快照，均不等于报告生成时再次访问服务器的状态；本生成器没有SSH或GPU访问。失败内容及原FAILED判定保持：

{chr(10).join(failures) if failures else '本次独立review未记录执行失败；这不构成精度或最终准入结论。'}

## 实际执行源码与来源

实际执行controller SHA256：`{summary['source_trace']['actual_executed_controller_sha256'] or '未归档'}`。派发记录与实际执行副本一致：`{summary['source_trace']['dispatch_matches_actual_executed_copy']}`。执行身份以上述实际副本为准；差异文件说明后续终态处理版的变化，不把本地较新源码冒作已执行版本。

{sources}

旧S32只提供来源与条件，既不重列被移除的旧吞吐数字，也不计算跨条件因果收益：

{chr(10).join(old_links)}

本次结果不包含模型精度复验或官方准入判定。
'''
    fragment = f'''<!-- 只供人工整理README；本生成器不修改v2提交文档。 -->
服务64／128复测已归档，配置上限与实际并发分别列出。{link(destination / '报告.md', '本次完整报告', destination)}；{link(summary['review_path'], '独立复核与原始来源', destination)}。

{capacity}

{performance}

{interpretations}

实际峰值仅取精确 `sglang:num_running_reqs` 原标签系列，未求和；范围见表，正式窗可用时前后各向内5秒，否则含预热。C128配置本身不证明128实际并发容量。KV预算、缓存大小、排队、同机双实例与5秒HTTP采样条件见完整报告。旧S32仅保留原始来源，不重列旧吞吐或推导因果收益。
'''
    return report, fragment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    review_path = args.review.resolve()
    require(review_path.is_file(), 'Independent review JSON is missing')
    require(not args.output_dir.exists() or args.output_dir.is_dir(), 'Output path is not a directory')
    for name in ('报告.md', 'readme-fragment.md', 'summary.json'):
        require(not (args.output_dir / name).exists(), 'Refusing to overwrite output: ' + name)
    review = json.loads(review_path.read_text())
    require(review.get('gpu_used') is False and isinstance(review.get('workers'), list),
            'Expected an independent CPU review')
    workers = [summarize_worker(row, review) for row in review['workers']]
    require(len(workers) == 2 and {row['client_concurrency'] for row in workers} == {64, 128},
            'Expected exactly the C64 and C128 independent review rows')
    summary = {'run_id': review.get('run_id'), 'status': review.get('status'),
               'review_path': str(review_path), 'review_sha256': sha(review_path),
               'source_trace': source_evidence(review, review_path), 'workers': workers,
               'gpu_used': False, 'ssh_used': False, 'original_files_modified': False,
               'default_submission_documents_modified': False,
               'old_S32_throughput_values_not_repeated': True, 'official_acceptance': 'NOT_ESTABLISHED',
               'accuracy': 'NOT_EVALUATED', 'comparison': 'Different conditions, side-by-side evidence only'}
    report, fragment = render(summary, args.output_dir.resolve())
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (('报告.md', report), ('readme-fragment.md', fragment), ('summary.json', encoded)):
        with (args.output_dir / name).open('x') as stream:
            stream.write(text)
    print(json.dumps({'status': 'REPORT_DRAFT_CREATED', 'output_dir': str(args.output_dir),
                      'files': ['报告.md', 'readme-fragment.md', 'summary.json'], 'gpu_used': False,
                      'submission_documents_modified': False}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'DRAFT_ERROR', 'error': str(error), 'gpu_used': False},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
