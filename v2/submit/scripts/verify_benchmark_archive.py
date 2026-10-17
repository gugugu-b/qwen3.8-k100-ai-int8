#!/usr/bin/env python3
"""Audit a sealed benchmark archive using only CPU and Python's standard library.

Original files and manifests are read only. Existing performance validators write
only inside a temporary directory, which is removed before this command exits.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def read_json(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def safe_relative(value, label):
    require(isinstance(value, str) and value and '\\' not in value
            and '\x00' not in value, 'Invalid relative path: ' + label)
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and '..' not in relative.parts
            and '.' not in relative.parts and relative.as_posix() == value
            and value != '.', 'Unsafe relative path: ' + label)
    return relative


def safe_file(root, value, label):
    relative = safe_relative(value, label)
    path = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), 'Symlink is not allowed: ' + label)
    require(path.resolve().is_relative_to(root), 'Path escapes archive: ' + label)
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0),
            'Invalid numeric metric: ' + label)
    return value


def close(actual, expected, label):
    number(actual, label)
    number(expected, label)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            'Metric differs: ' + label + ' (' + str(actual) + ' versus ' + str(expected) + ')')


def compare(actual, expected, label):
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(set(actual) == set(expected), 'Report keys differ: ' + label)
        for key in actual:
            compare(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(actual, list) and isinstance(expected, list):
        require(len(actual) == len(expected), 'Report array length differs: ' + label)
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, label + '/' + str(index))
    elif (isinstance(actual, (int, float)) and not isinstance(actual, bool)
          and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        # Saved timing ranges and other report numbers may contain negative
        # diagnostic changes, so compare signed finite values here.
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                'Report numeric value differs: ' + label)
    else:
        require(actual == expected, 'Report value differs: ' + label)


def pair(value):
    if isinstance(value, str):
        value = [int(part) for part in value.split(',')]
    require(isinstance(value, (list, tuple)) and list(value) in
            ([0, 1], [2, 3], [4, 5], [6, 7]), 'Invalid physical TP2 pair')
    return list(value)


def run_validator(arguments, cwd):
    result = subprocess.run([sys.executable, '-I', *map(str, arguments)],
                            cwd=cwd, text=True, capture_output=True)
    require(result.returncode == 0,
            'CPU validator failed: ' + Path(arguments[0]).name + ': '
            + (result.stderr or result.stdout)[-1500:])


def fixed_random(directory, package, temporary):
    require(sorted(p.name for p in directory.glob('round-*') if p.is_dir()) ==
            ['round-01', 'round-02', 'round-03'], 'Expected exactly three formal random rounds')
    report_file = temporary / 'random-review.json'
    run_validator([package / 'scripts/review_performance.py', directory,
                   '--out', report_file], temporary)
    reviewed = read_json(report_file)
    compare(reviewed, read_json(directory / 'reference_performance_review.json'),
            str(directory) + '/reference_performance_review.json')
    rounds = [read_json(directory / ('round-%02d' % i) / 'summary.json')
              for i in (1, 2, 3)]
    for i, row in enumerate(rounds, 1):
        require(row['phase'] == 'measure' and row['round'] == i,
                'A non-formal/random round was included')
        number(row['output_throughput_tps'], 'random output TPS', positive=True)
    cfg = read_json(directory / 'config.json')
    return {'tps': [row['output_throughput_tps'] for row in rounds],
            'ttft': statistics.mean(row['ttft_s']['mean'] for row in rounds),
            'tpot': statistics.mean(row['tpot_ms']['mean'] for row in rounds),
            'concurrency': cfg['concurrency'], 'input_len': cfg['input_len'],
            'text_flags': len(reviewed['text_flags'])}


def sharegpt(directory, package, temporary):
    files = ['round-01.jsonl', 'round-02.jsonl', 'round-03.jsonl']
    require(sorted(p.name for p in directory.glob('round-*.jsonl')) == files,
            'Expected exactly three formal ShareGPT rounds')
    require(sorted(p.name for p in directory.glob('full-warmup-*.jsonl')) ==
            ['full-warmup-01.jsonl'], 'Expected one complete ShareGPT warmup')
    copied = temporary / 'sharegpt'
    copied.mkdir()
    for name in files + ['full-warmup-01.jsonl']:
        shutil.copyfile(directory / name, copied / name)
    run_validator([package / 'scripts/validate_sharegpt.py', copied,
                   '--rounds', '3', '--full-warmup-rounds', '1'], temporary)
    compare(read_json(copied / 'validated_summary.json'),
            read_json(directory / 'validated_summary.json'),
            str(directory) + '/validated_summary.json')
    # The unmodified SGLang output serializes request_rate=inf as Infinity.
    # Keep that original representation; all performance numbers used below
    # are checked for finiteness separately.
    rounds = [json.loads((directory / name).read_text()) for name in files]
    for row in rounds:
        number(row['output_throughput'], 'ShareGPT output TPS', positive=True)
        require(row['total_output_tokens'] == sum(row['output_lens']),
                'ShareGPT output token totals differ')
        close(row['output_throughput'], row['total_output_tokens'] / row['duration'],
              'ShareGPT output tokens / formal duration')
    return {'tps': [row['output_throughput'] for row in rounds],
            'ttft': statistics.mean(row['mean_ttft_ms'] for row in rounds) / 1000,
            'tpot': statistics.mean(row['mean_tpot_ms'] for row in rounds),
            'concurrency': rounds[0]['max_concurrency'], 'text_flags': 0}


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
                        default=package / 'logs/v2_benchmark_raw/final')
    args = parser.parse_args()
    archive = args.archive.resolve()
    require(archive.is_dir(), 'Archive directory does not exist')
    require(not (archive / 'manifest.json').is_symlink(), 'Manifest cannot be a symlink')
    manifest = read_json(archive / 'manifest.json')
    require(manifest['format_version'] == 1, 'Unsupported archive format')
    files = manifest['files']
    require(isinstance(files, dict) and files, 'Manifest files must be a nonempty mapping')
    require('manifest.json' not in files, 'Manifest must not hash itself')
    observed = set()
    for parent, directories, names in os.walk(archive, followlinks=False):
        for name in directories:
            require(not (Path(parent) / name).is_symlink(), 'Archive directory symlink detected')
        for name in names:
            path = Path(parent) / name
            require(not path.is_symlink() and path.is_file(), 'Archive contains a non-regular file')
            relative = path.relative_to(archive).as_posix()
            if relative != 'manifest.json':
                observed.add(relative)
    require(observed == set(files), 'Manifest file coverage differs from archive contents')
    for name, entry in files.items():
        path = safe_file(archive, name, name)
        require(isinstance(entry['bytes'], int) and entry['bytes'] >= 0,
                'Invalid file size in manifest: ' + name)
        require(isinstance(entry['sha256'], str) and len(entry['sha256']) == 64,
                'Invalid SHA256 field: ' + name)
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Size/SHA256 mismatch: ' + name)
        if entry.get('source_package_path') is not None:
            safe_relative(entry['source_package_path'], 'source_package_path:' + name)
    summary = read_json(safe_file(archive, 'summary.json', 'summary.json'))
    scenes = manifest['scenes']
    require(isinstance(scenes, list) and scenes, 'No scenes in archive')
    summary_rows = summary['scenes']
    require(len({row['id'] for row in summary_rows}) == len(summary_rows),
            'Duplicate scene IDs in summary')
    by_id = {row['id']: row for row in summary_rows}
    require(len({row['id'] for row in scenes}) == len(scenes) and
            len({row['directory'] for row in scenes}) == len(scenes),
            'Duplicate scene IDs or directories in manifest')
    require(set(by_id) == {row['id'] for row in scenes}, 'Summary scene coverage differs')
    random_primary = set()
    sharegpt_primary = set()
    high_client_diagnostics = set()
    flagged = 0
    for scene in scenes:
        require(scene['kind'] in ('fixed_random', 'sharegpt'), 'Unknown scene kind')
        require(scene['role'] in ('primary_random', 'primary_sharegpt', 'sla', 'clean_runtime',
                                  'diagnostic_high_client_concurrency'),
                'Unknown scene role')
        require(isinstance(scene['concurrency'], int) and not isinstance(scene['concurrency'], bool)
                and scene['concurrency'] > 0, 'Invalid concurrency')
        safe_relative(scene['source_directory'], 'source_directory:' + scene['id'])
        directory = safe_file(archive, scene['directory'], scene['id'])
        require(directory.is_dir(), 'Missing scene directory: ' + scene['id'])
        row = by_id[scene['id']]
        for key in ('directory', 'kind', 'role', 'concurrency'):
            require(row[key] == scene[key], 'Scene metadata differs: ' + scene['id'] + '/' + key)
        require(pair(row['physical_pair']) == pair(scene['physical_pair']),
                'Scene physical pair differs: ' + scene['id'])
        with tempfile.TemporaryDirectory(prefix='qwen38-archive-review-') as path:
            temporary = Path(path)
            result = (fixed_random(directory, package, temporary)
                      if scene['kind'] == 'fixed_random' else sharegpt(directory, package, temporary))
        require(result['concurrency'] == scene['concurrency'], 'Raw concurrency differs: ' + scene['id'])
        require(len(row['output_tps_rounds']) == len(result['tps']) == 3,
                'Three formal TPS values required: ' + scene['id'])
        for i, (saved, actual) in enumerate(zip(row['output_tps_rounds'], result['tps']), 1):
            close(saved, actual, scene['id'] + '/formal-round-' + str(i))
        close(row['mean_output_tps'], statistics.mean(result['tps']), scene['id'] + '/mean_output_tps')
        close(row['ttft_mean_s'], result['ttft'], scene['id'] + '/ttft_mean_s')
        close(row['tpot_mean_ms'], result['tpot'], scene['id'] + '/tpot_mean_ms')
        flagged += bool(result['text_flags'])
        if scene['role'] == 'primary_random':
            require(scene['kind'] == 'fixed_random', 'Primary random has wrong kind')
            key = (result['input_len'], scene['concurrency'])
            require(key not in random_primary, 'Duplicate primary random workload')
            random_primary.add(key)
        if scene['role'] == 'primary_sharegpt':
            require(scene['kind'] == 'sharegpt', 'Primary ShareGPT has wrong kind')
            require(scene['concurrency'] not in sharegpt_primary, 'Duplicate primary ShareGPT concurrency')
            sharegpt_primary.add(scene['concurrency'])
        if scene['role'] == 'diagnostic_high_client_concurrency':
            require(scene['kind'] == 'fixed_random', 'High-client diagnostic has wrong kind')
            key = (result['input_len'], scene['concurrency'])
            require(key not in high_client_diagnostics, 'Duplicate high-client diagnostic workload')
            high_client_diagnostics.add(key)
    expected_random = {(4096, c) for c in (1, 8, 16, 32)} | {
        (65536, c) for c in (1, 8, 16, 32)}
    require(random_primary == expected_random, 'The eight primary random workloads are incomplete')
    require(sharegpt_primary == {1, 4, 16, 32, 64}, 'The five primary ShareGPT workloads are incomplete')
    require(high_client_diagnostics == {(4096, 64), (4096, 128)},
            'The two high-client diagnostic workloads are incomplete')
    require(summary['primary_random_count'] == 8 and summary['primary_sharegpt_count'] == 5,
            'Summary primary counts differ')
    if 'diagnostic_high_client_concurrency_count' in summary:
        require(summary['diagnostic_high_client_concurrency_count'] == 2,
                'Summary high-client diagnostic count differs')
    print(json.dumps({'status': 'PASS', 'files': len(files), 'scenes': len(scenes),
                      'formal_rounds': len(scenes) * 3, 'primary_random_count': 8,
                      'primary_sharegpt_count': 5, 'text_flagged_scenes': flagged,
                      'diagnostic_high_client_concurrency_count': 2,
                      'scope': 'Archive integrity and saved performance arithmetic; text flags retained',
                      'gpu_used': False}, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'FAIL', 'error': str(error), 'gpu_used': False},
                         ensure_ascii=False, separators=(',', ':')), file=sys.stderr)
        raise SystemExit(1)
