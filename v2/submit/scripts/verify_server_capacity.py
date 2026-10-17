#!/usr/bin/env python3
"""CPU-only review of collected server64/Graph64 and server128/Graph128 runs.

The collection and all original reports are read only. The reference auditor
writes only a temporary report. Metrics peaks include benchmark warmup and are
kept separately for every original metric name and exact label string.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


WORKSPACE = Path(__file__).resolve().parents[2]
REMOTE_BASE = PurePosixPath('/data/wangzy/v2')
IMAGE = 'sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
EXPECTED = ({'key': '45', 'pair': '4,5', 'c': 64},
            {'key': '67', 'pair': '6,7', 'c': 128})
METRIC_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value, 'Unsafe relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == value
            and value != '.', 'Unsafe relative path: ' + value)
    return path


def local_path(collection, relative):
    path = safe_relative(relative)
    result = collection.joinpath(*path.parts)
    require(result.resolve().is_relative_to(collection), 'Collection path escapes root')
    cursor = collection
    for part in path.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'Symlink evidence path is not accepted')
    return result


def from_remote(collection, remote):
    path = PurePosixPath(remote)
    require(path.is_absolute() and '..' not in path.parts, 'Unsafe remote evidence path')
    return local_path(collection, path.relative_to(REMOTE_BASE).as_posix())


def compare(actual, expected, location):
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(set(actual) == set(expected), 'Report keys differ: ' + location)
        for key in actual:
            compare(actual[key], expected[key], location + '/' + str(key))
    elif isinstance(actual, list) and isinstance(expected, list):
        require(len(actual) == len(expected), 'Report array differs: ' + location)
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, location + '/' + str(index))
    elif (isinstance(actual, (int, float)) and not isinstance(actual, bool)
          and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                'Report numeric value differs: ' + location)
    else:
        require(actual == expected, 'Report value differs: ' + location)


def metrics_peaks(path, requested_concurrency, windows=None):
    result = {'status': 'NOT_COLLECTED', 'path': str(path), 'raw_samples': 0,
              'http_success_samples': 0, 'error_samples': 0, 'parse_errors': 0,
              'nonfinite_candidate_values': 0, 'errors': [], 'series': [],
              'scope': 'BENCHMARK_INCLUDING_WARMUP; NO_ABSOLUTE_FORMAL_WINDOW_FILTER',
              'global_running_request_peak': None, 'aggregation': 'MAX_OVER_SAMPLES_PER_EXACT_NAME_AND_LABELS; NEVER_SUM',
              'requested_client_concurrency': requested_concurrency}
    if windows:
        result.update(scope='FORMAL_ROUND_WINDOW_INTERIORS; PRE_GET_UTC; FIVE_SECOND_INWARD_MARGIN',
                      selection_windows=windows, outside_window_samples=0, samples_without_usable_utc=0)
    if not path.is_file():
        return result
    result.update(status='RAW_SAMPLES_PARSED', bytes=path.stat().st_size, sha256=sha(path))
    peaks = {}
    untyped = set()
    with path.open() as stream:
        for index, line in enumerate(stream, 1):
            result['raw_samples'] += 1
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                result['parse_errors'] += 1
                if len(result['errors']) < 20:
                    result['errors'].append({'sample': index, 'error': repr(error)})
                continue
            if windows:
                try:
                    timestamp = datetime.fromisoformat(row['utc'].replace('Z', '+00:00')).timestamp()
                except (ValueError, KeyError, TypeError):
                    result['samples_without_usable_utc'] += 1
                    continue
                if not any(window['safe_start_epoch'] <= timestamp <= window['safe_end_epoch']
                           for window in windows):
                    result['outside_window_samples'] += 1
                    continue
            if row.get('error') or row.get('status_code') != 200 or not isinstance(row.get('text'), str):
                result['error_samples'] += 1
                if len(result['errors']) < 20:
                    result['errors'].append({'sample': index, 'utc': row.get('utc'),
                                             'error': row.get('error'), 'status_code': row.get('status_code')})
                continue
            result['http_success_samples'] += 1
            text = row['text']
            types = {}
            for sample_line in text.splitlines():
                if sample_line.startswith('# TYPE '):
                    fields = sample_line.split()
                    if len(fields) == 4:
                        types[fields[2]] = fields[3]
            for sample_line in text.splitlines():
                if not sample_line or sample_line.startswith('#'):
                    continue
                match = METRIC_LINE.match(sample_line)
                if not match:
                    continue
                name, labels, raw_value = match.groups()
                lowered = name.lower()
                running = 'running' in lowered and ('req' in lowered or 'request' in lowered)
                queue = any(word in lowered for word in ('queue', 'waiting', 'pending'))
                if not (running or queue):
                    continue
                if types.get(name) != 'gauge':
                    untyped.add((name, types.get(name, 'NOT_DECLARED')))
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    result['parse_errors'] += 1
                    continue
                if not math.isfinite(value):
                    result['nonfinite_candidate_values'] += 1
                    continue
                labels = labels or ''
                key = (name, labels)
                entry = peaks.get(key)
                if entry is None:
                    entry = dict(metric_name=name, original_labels=labels,
                                 category='running_requests' if running else 'queue_gauge',
                                 peak_value=value, observations=0, peak_sample=index,
                                 peak_utc=row.get('utc'), original_peak_line=sample_line)
                    peaks[key] = entry
                entry['observations'] += 1
                if value > entry['peak_value']:
                    entry.update(peak_value=value, peak_sample=index,
                                 peak_utc=row.get('utc'), original_peak_line=sample_line)
    result['series'] = [peaks[key] for key in sorted(peaks)]
    result['candidate_metrics_without_gauge_declaration'] = [
        {'metric_name': name, 'declared_type': type_} for name, type_ in sorted(untyped)]
    running_series = [entry for entry in result['series'] if entry['category'] == 'running_requests']
    result['individual_running_series_reaching_requested_C'] = [
        {'metric_name': entry['metric_name'], 'original_labels': entry['original_labels'],
         'peak_value': entry['peak_value']} for entry in running_series
        if entry['peak_value'] >= requested_concurrency]
    result['configured_C_is_not_proof_of_actual_running_C'] = True
    return result


def formal_windows(observer, key, directory):
    result = {'status': 'NOT_AVAILABLE', 'observer': str(observer), 'windows': [],
              'inward_margin_s': 5, 'required_rounds': [1, 2, 3],
              'clock_basis': 'Benchmark started_at epoch and sampler UTC on the same server',
              'sample_timestamp': 'UTC recorded before HTTP GET; 4s configured timeout'}
    if not observer.is_file():
        return result
    starts = {}
    snapshots = 0
    parse_errors = 0
    with observer.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
                sample = value.get('samples', {}).get(key)
                if not isinstance(sample, dict) or sample.get('phase') != 'measure':
                    continue
                index = sample.get('round')
                epoch = sample.get('started_at')
                if index not in (1, 2, 3) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
                    continue
                if index in starts:
                    require(math.isclose(starts[index], epoch, rel_tol=0, abs_tol=1e-6),
                            'Observer recorded conflicting started_at for a formal round')
                starts[index] = epoch
                snapshots += 1
            except (ValueError, TypeError) as error:
                if 'conflicting started_at' in str(error):
                    raise
                parse_errors += 1
    result.update(observer_sha256=sha(observer), observed_rounds=sorted(starts),
                  formal_progress_snapshots=snapshots, parse_errors=parse_errors)
    if set(starts) != {1, 2, 3}:
        result['status'] = 'INCOMPLETE_WINDOW_EVIDENCE; NO_FORMAL_ONLY_PEAK'
        return result
    for index in (1, 2, 3):
        summary = directory / f'round-{index:02d}/summary.json'
        if not summary.is_file():
            result.update(status='MISSING_FORMAL_ROUND_SUMMARY; NO_FORMAL_ONLY_PEAK', windows=[])
            return result
        row = read_json(summary)
        duration = row.get('wall_time_s')
        require(row.get('phase') == 'measure' and row.get('round') == index
                and isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 10,
                'Invalid formal window summary')
        start = starts[index]
        result['windows'].append({'round': index, 'started_at_epoch': start,
                                  'wall_time_s': duration, 'ended_at_epoch': start + duration,
                                  'safe_start_epoch': start + 5, 'safe_end_epoch': start + duration - 5,
                                  'summary_path': str(summary), 'summary_sha256': sha(summary)})
    result['status'] = 'ALL_THREE_FORMAL_WINDOWS_RECORDED'
    return result


def review_scene(directory, package, concurrency):
    report = {'directory': str(directory), 'status': 'INCOMPLETE', 'three_round_mean': None}
    files = sorted(directory.glob('round-*/summary.json')) if directory.is_dir() else []
    report['available_formal_round_files'] = [str(path.relative_to(directory)) for path in files]
    saved = directory / 'reference_performance_review.json'
    if (len(files) != 3 or len(list(directory.glob('warmup-*/summary.json'))) != 2
            or not saved.is_file()):
        report['reason'] = 'Complete three formal rounds, two warmups and original review are required'
        return report
    auditor = package / 'scripts/review_performance.py'
    with tempfile.TemporaryDirectory(prefix='qwen38-high-capacity-review-') as temporary:
        output = Path(temporary) / 'reference-review.json'
        process = subprocess.run([sys.executable, '-I', str(auditor), str(directory),
                                  '--out', str(output)], text=True, capture_output=True)
        require(process.returncode == 0, 'Independent raw audit failed: '
                + (process.stderr or process.stdout)[-2000:])
        reproduced = read_json(output)
    compare(reproduced, read_json(saved), 'reference_performance_review.json')
    cfg = read_json(directory / 'config.json')
    require(cfg['concurrency'] == concurrency and cfg['input_len'] == 4096
            and cfg['output_len'] == 1024 and cfg['warmup_rounds'] == 2
            and cfg['measure_rounds'] == 3 and cfg['requests_per_round'] == 2 * concurrency,
            'Actual benchmark workload differs from requested capacity test')
    rounds = [read_json(path) for path in files]
    require([row['round'] for row in rounds] == [1, 2, 3]
            and all(row['phase'] == 'measure' for row in rounds), 'Formal round identifiers differ')
    values = [row['output_throughput_tps'] for row in rounds]
    flags = reproduced['text_flags']
    means = {'output_tps': statistics.mean(values)}
    for metric in ('ttft_s', 'tpot_ms', 'e2e_s'):
        for stat in ('mean', 'p95', 'p99'):
            means[metric + '_' + stat] = statistics.mean(row[metric][stat] for row in rounds)
    report.update(status='PERFORMANCE_RAW_REVIEW_REPRODUCED', formal_rounds=3,
                  formal_requests=6 * concurrency, output_tps_rounds=values,
                  three_round_mean=means, original_performance_status=reproduced['performance_status'],
                  original_text_integrity_status=reproduced['text_integrity_status'],
                  flags_formal=[row for row in flags if row['phase'] == 'measure'],
                  flags_warmup=[row for row in flags if row['phase'] == 'warmup'],
                  round_metrics=[{'round': row['round'], 'output_tps': row['output_throughput_tps'],
                                  'ttft_s': row['ttft_s'], 'tpot_ms': row['tpot_ms'],
                                  'e2e_s': row['e2e_s']} for row in rounds],
                  original_review_sha256=sha(saved), auditor_path=str(auditor),
                  auditor_sha256=sha(auditor), config=cfg,
                  source_files_sha256=reproduced['raw_sha256'],
                  quantiles='Mean of three per-round quantiles; not pooled quantiles',
                  accuracy='NOT_EVALUATED_BY_PERFORMANCE_AUDIT')
    return report


def previous_server32(package, concurrency):
    directory = package / f'result/runs/community-official-20261017-c-p45-matrix-4k-c{concurrency}/4k-c{concurrency}'
    source = directory / 'reference_performance_review.json'
    if not source.is_file():
        return {'status': 'OLD_S32_RECORD_NOT_AVAILABLE', 'path': str(source)}
    original = read_json(source)
    info = read_json(directory / 'server-info.json')
    args = info.get('server_args', info)
    return {'status': 'HISTORICAL_SIDE_BY_SIDE_ONLY', 'path': str(source), 'sha256': sha(source),
            'physical_pair': '4,5', 'client_concurrency': concurrency,
            'server_max_running_requests': args.get('max_running_requests'),
            'cuda_graph_max_bs': args.get('cuda_graph_max_bs'),
            'output_tps_rounds': original['output_tps_rounds'],
            'mean_output_tps': original['mean_output_tps'],
            'ttft_mean_s': original['ttft_s_mean_mean_of_rounds'],
            'tpot_mean_ms': original['tpot_ms_mean_mean_of_rounds'],
            'text_integrity_status': original['text_integrity_status'],
            'comparison': 'Different server/Graph limits, date and host activity; C128 also changes pair. '
                          'No contemporaneous single-variable causal comparison or improvement percentage.'}


def ownership_review(path, run_id, pair, require_stopped=False):
    if not path.is_file():
        return {'status': 'NOT_AVAILABLE', 'path': str(path)}
    data = read_json(path)
    labels = data['Config'].get('Labels', {})
    require(data['Image'] == IMAGE and labels.get('qwen38.community.run') == run_id
            and labels.get('qwen38.community.pair') == pair
            and f'HIP_VISIBLE_DEVICES={pair}' in data['Config']['Env'], 'Container identity mismatch')
    if require_stopped:
        require(data['State']['Running'] is False, 'Owned container was not stopped')
    return {'status': 'OWNED_IDENTITY_VERIFIED', 'path': str(path), 'sha256': sha(path),
            'container_id': data.get('Id'), 'running': data['State']['Running'],
            'image': data['Image'], 'run_label': run_id, 'physical_pair': pair}


def capacity_markers(paths):
    markers = []
    oom = False
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(errors='replace')
        for line in text.splitlines():
            lowered = line.lower()
            is_oom = any(term in lowered for term in ('out of memory', 'outofmemoryerror', 'out_of_memory'))
            if is_oom or any(term in lowered for term in ('allocation failed', 'traceback', 'hip error', 'cuda error')):
                oom = oom or is_oom
                if len(markers) < 40:
                    markers.append({'path': str(path), 'line': line[:1500]})
    return {'oom_marker_observed': oom, 'log_markers': markers,
            'interpretation': 'Observed log text; no automatic downgrade or unproven root-cause inference'}


def review_worker(collection, control, package, expected, run_id):
    key, pair, concurrency = expected['key'], expected['pair'], expected['c']
    statefile = control / f'p{key}/state.json'
    result = dict(pair=pair, requested_client_concurrency=concurrency,
                  requested_server_max_running_requests=concurrency,
                  requested_cuda_graph_max_bs=concurrency, review_status='INCOMPLETE', issues=[])
    if not statefile.is_file():
        result['reason'] = 'Worker state is not collected'
        return result
    state = read_json(statefile)
    result.update(worker_status=state.get('status'), worker_state_path=str(statefile),
                  worker_state_sha256=sha(statefile), error=state.get('error'),
                  traceback=state.get('traceback'), outcome_before_cleanup=state.get('outcome_before_cleanup'),
                  cleanup_status=state.get('cleanup_status'), cleanup_error=state.get('cleanup_error'),
                  lock_acquired_reported=state.get('lock_acquired'),
                  lock_released_reported=state.get('lock_released'), gpu_after=state.get('gpu_after'))
    gpu_after = state.get('gpu_after')
    try:
        result['gpu_after_snapshot_idle'] = bool(gpu_after) and all(
            float(values['HCU use (%)']) == 0
            and float(values['vram Total Used Memory (MiB)']) <= 16 for values in gpu_after.values())
    except (AttributeError, KeyError, ValueError, TypeError):
        result['gpu_after_snapshot_idle'] = None
    def inspect_check(name, action):
        try:
            result[name] = action()
        except Exception as error:
            result['issues'].append({'check': name, 'error': str(error)})
            result[name] = {'status': 'VERIFICATION_FAILED', 'error': str(error)}
    require(state.get('pair') == pair and state.get('client_concurrency') == concurrency,
            'Worker state pair or requested concurrency differs')
    require(state.get('server_max_running_requests') == concurrency
            and state.get('cuda_graph_max_bs') == concurrency, 'Requested capacity state differs')
    target = control / ('runtime-p' + key)
    api = target / 'server-info.json'
    if api.is_file():
        def check_api():
            info = read_json(api)
            args = info.get('server_args', info)
            expected_args = dict(tp_size=2, pp_size=1, max_running_requests=concurrency,
                                 cuda_graph_max_bs=concurrency, disable_cuda_graph=False,
                                 disable_custom_all_reduce=False, custom_all_reduce_backend='native',
                                 context_length=67584, page_size=64, mem_fraction_static=0.85,
                                 max_prefill_tokens=65536, chunked_prefill_size=8192,
                                 disable_radix_cache=True, mamba_scheduler_strategy='no_buffer',
                                 attention_backend='fa3', mm_attention_backend='fa3',
                                 speculative_num_steps=1, speculative_eagle_topk=1,
                                 speculative_num_draft_tokens=2)
            for name, value in expected_args.items():
                require(args.get(name) == value, 'Actual API parameter differs: ' + name)
            require(args.get('speculative_algorithm') in ('NEXTN', 'EAGLE'), 'Actual MTP algorithm differs')
            require(max(args.get('cuda_graph_bs', [0])) == concurrency, 'Actual graph list does not reach C')
            return {'status': 'ACTUAL_API_CAPACITY_VERIFIED', 'path': str(api), 'sha256': sha(api),
                    'actual': {name: args.get(name) for name in expected_args},
                    'cuda_graph_bs': args['cuda_graph_bs'],
                    'does_not_prove_actual_running_request_peak': True}
        inspect_check('api_capacity', check_api)
    else:
        result['api_capacity'] = {'status': 'NOT_AVAILABLE; STARTUP_MAY_NOT_HAVE_COMPLETED'}
    gate = target / 'gate/review.json'
    if gate.is_file():
        def check_gate():
            review = read_json(gate)
            require(review['status'] == 'RUNTIME_PAIR_NUMERICAL_GRAPH_PASS'
                    and review['checks'] == 162 and review['exclusions'] == 20
                    and review['pair'] == pair, 'Numerical/Graph gate differs')
            return {'status': review['status'], 'path': str(gate), 'sha256': sha(gate),
                    'checks': 162, 'exclusions': 20, 'actual_model_graph_gate': 'REVIEW_REQUIRED'}
        inspect_check('runtime_gate', check_gate)
    else:
        result['runtime_gate'] = {'status': 'NOT_AVAILABLE'}
    inspect_check('container_initial', lambda: ownership_review(target / 'container-inspect.json', run_id, pair))
    inspect_check('container_stopped', lambda: ownership_review(
        statefile.parent / 'container-stopped.json', run_id, pair, require_stopped=True))
    result['metrics'] = metrics_peaks(target / 'running-metrics.jsonl', concurrency)
    result['previous_server32'] = previous_server32(package, concurrency)
    server_run = state.get('server_run_id', f'{run_id}-p{key}-server')
    server_log = local_path(collection, 'submit/logs/' + server_run + '/server.log')
    if not server_log.is_file():
        server_log = target / 'server-final/server.log'
    if server_log.is_file():
        log = server_log.read_text(errors='replace')
        result['native_guard_eligible_ranks'] = [rank for rank in (0, 1)
            if f'[Native BF16 AR] Eligible rank={rank} world_size=2' in log]
    result['capacity_failure_evidence'] = capacity_markers(
        [server_log, statefile.parent / 'startup.log', statefile.parent / f'server{concurrency}-4k-c{concurrency}.log'])
    scenes = state.get('scenes', [])
    candidates = [from_remote(collection, row['path']) for row in scenes if row.get('path')]
    if not candidates:
        candidates = sorted((collection / 'submit/result/runs').glob(
            f'{run_id}-p{key}-server{concurrency}-4k-c{concurrency}/4k-c{concurrency}'))
    result['scenes'] = []
    for directory in candidates:
        try:
            result['scenes'].append(review_scene(directory, package, concurrency))
        except Exception as error:
            result['issues'].append({'check': 'independent_raw_review', 'error': str(error)})
            result['scenes'].append({'directory': str(directory), 'status': 'RAW_REVIEW_FAILED', 'error': str(error)})
    if len(candidates) == 1:
        try:
            windows = formal_windows(control / 'progress-observer.jsonl', key, candidates[0])
            result['formal_window_evidence'] = windows
            if windows['status'] == 'ALL_THREE_FORMAL_WINDOWS_RECORDED':
                result['formal_window_metrics'] = metrics_peaks(
                    target / 'running-metrics.jsonl', concurrency, windows['windows'])
        except Exception as error:
            result['formal_window_evidence'] = {'status': 'WINDOW_REVIEW_FAILED; NO_FORMAL_ONLY_PEAK',
                                               'error': str(error)}
    complete = (len(result['scenes']) == 1 and result['scenes'][0]['status'] == 'PERFORMANCE_RAW_REVIEW_REPRODUCED')
    terminal_success = state.get('status') == 'BENCHMARK_COMPLETE'
    if terminal_success:
        try:
            require(complete, 'Worker claims completion without one independently complete scene')
            require(state.get('lock_acquired') is True and state.get('lock_released') is True,
                    'Completed worker lacks acquired/released lock evidence')
            for field in ('api_capacity', 'runtime_gate', 'container_stopped'):
                require(result[field]['status'] not in ('NOT_AVAILABLE', 'VERIFICATION_FAILED')
                        and 'NOT_AVAILABLE' not in result[field]['status'], 'Completed worker lacks ' + field)
            require(result.get('native_guard_eligible_ranks') == [0, 1], 'Both model native guard ranks are required')
        except ValueError as error:
            result['issues'].append({'check': 'completion_evidence', 'error': str(error)})
        result['review_status'] = 'COMPLETE_PERFORMANCE_RECHECKED' if not result['issues'] else 'VERIFICATION_HAS_ISSUES'
    elif state.get('status') in ('FAILED', 'CLEANUP_FAILED'):
        result['review_status'] = 'RUN_FAILED_WITH_EVIDENCE'
    result['formal_only_global_running_request_peak'] = None
    result['resource_release_status'] = ('REPORTED_LOCK_RELEASE_AND_OWNED_STOP_VERIFIED'
        if state.get('lock_released') is True
        and result['container_stopped']['status'] == 'OWNED_IDENTITY_VERIFIED'
        else 'RESOURCE_RELEASE_NOT_FULLY_VERIFIED')
    result['accuracy_or_official_acceptance'] = 'NOT_ESTABLISHED'
    return result


def verify_collection_inventory(collection):
    source = collection.parent / 'server-source-inventory.json'
    if not source.is_file():
        return {'status': 'NOT_PROVIDED', 'path': str(source)}
    manifest = read_json(source)
    count = 0
    for name, entry in manifest['files'].items():
        path = local_path(collection, name)
        require(path.is_file() and path.stat().st_size == entry['bytes']
                and sha(path) == entry['sha256'], 'Collected source identity differs: ' + name)
        count += 1
    return {'status': 'ALL_SOURCE_FILES_SIZE_SHA256_PASS', 'files': count,
            'path': str(source), 'sha256': sha(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-base', type=Path, default=WORKSPACE)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite review output')
    collection = args.collection_root.resolve()
    package = args.package.resolve()
    require(collection.is_dir() and (package / 'scripts/review_performance.py').is_file(),
            'Collection or local reference auditor is missing')
    controls = [path for path in collection.glob('retest-server64-128-*')
                if path.is_dir() and (path / 'controller-source.py').is_file()]
    require(len(controls) == 1, 'Expected exactly one collected retest control directory')
    control = controls[0]
    plan = read_json(control / 'execution-plan.json')
    run_id = plan['run_id']
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', run_id) is not None, 'Unsafe collected run ID')
    result = {'reviewed_utc': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
              'collection_root': str(collection), 'control': str(control),
              'source_base': str(args.source_base.resolve()), 'gpu_used': False,
              'ssh_used': False, 'original_files_modified': False, 'workers': [],
              'comparison_scope': 'Cross-condition side-by-side only; not causal A/B',
              'metrics_scope': 'All benchmark peaks include warmup. Optional formal_window_metrics '
                               'uses all three observer windows with 5s inward margins and pre-GET UTC.',
              'submission_ready': False}
    result['collection_identity'] = verify_collection_inventory(collection)
    for name in ('controller-source.py', 'finished.json', 'controller-failure.json'):
        path = control / name
        if path.is_file():
            result[name] = {'path': str(path), 'sha256': sha(path)}
    for expected in EXPECTED:
        try:
            worker = review_worker(collection, control, package, expected, run_id)
        except Exception as error:
            worker = {'pair': expected['pair'], 'requested_client_concurrency': expected['c'],
                      'review_status': 'VERIFICATION_FAILED', 'error': str(error), 'gpu_used': False}
        result['workers'].append(worker)
    result['status'] = ('COMPLETE_PERFORMANCE_RECHECKED' if all(
        row['review_status'] == 'COMPLETE_PERFORMANCE_RECHECKED' for row in result['workers'])
        else 'HAS_FAILURES_OR_INCOMPLETE_EVIDENCE')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'workers': [{'pair': row['pair'], 'review_status': row['review_status']}
                                  for row in result['workers']], 'gpu_used': False},
                     ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'REVIEW_ERROR', 'error': str(error), 'gpu_used': False},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
