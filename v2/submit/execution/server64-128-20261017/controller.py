#!/usr/bin/env python3
"""Owned TP2 C64/server64/Graph64 and C128/server128/Graph128 retests.

Use --dry-check for CPU-only source checks. Only --run can start containers.
The declared default candidate and deployment scripts are never rewritten.
"""
import argparse
import ast
import concurrent.futures
import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
from pathlib import Path


ROOT = Path('/data/wangzy/v2/submit')
DEFAULT_CONTROL = Path('/data/wangzy/v2/retest-server64-128-20261017-a')
DEFAULT_RUN = 'high-concurrency-20261017-a'
WORKERS = ({'pair': '4,5', 'port': 31540, 'c': 64},
           {'pair': '6,7', 'port': 31560, 'c': 128})
IMAGE = 'sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
SITE_SHA = 'da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_core(package, control, run_id):
    source = package / 'execution/official-alignment-20261017-c/controller.py'
    module = types.ModuleType('qwen38_reference_helpers')
    module.__file__ = str(source)
    # exec(compiled source) avoids writing __pycache__ into the sealed package.
    exec(compile(source.read_text(), str(source), 'exec'), module.__dict__)
    module.PACKAGE = package
    module.SOURCE = package / 'execution/official-alignment-20261017-c'
    module.CONTROL = control
    module.RUN = run_id
    module.LIBRARY = package / 'config/library_gpu4_int8tuned_stock_clean'
    require(module.IMAGE == IMAGE and module.SITE_SHA == SITE_SHA, 'Reference identity differs')
    return module


def private_script_texts(package, concurrency):
    require(concurrency in (64, 128), 'Unsupported retest concurrency')
    serve = (package / 'scripts/serve.sh').read_text()
    start = (package / 'scripts/start_server.sh').read_text()
    serve_anchor = 'PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)'
    start_anchor = 'PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)'
    expected = [(serve_anchor, serve), (start_anchor, start),
                ('--max-running-requests 32', serve), ('--cuda-graph-max-bs 32', serve)]
    for needle, source in expected:
        require(source.count(needle) == 1, 'Deployment anchor differs: ' + needle)
    assignment = 'PACKAGE_DIR=' + shlex.quote(str(package))
    changed_serve = serve.replace(serve_anchor, assignment)
    changed_serve = changed_serve.replace('--max-running-requests 32',
                                          '--max-running-requests ' + str(concurrency))
    changed_serve = changed_serve.replace('--cuda-graph-max-bs 32',
                                          '--cuda-graph-max-bs ' + str(concurrency))
    changed_start = start.replace(start_anchor, assignment)
    # A reversible textual check proves no unrelated service argument changed.
    restored = changed_serve.replace(assignment, serve_anchor, 1)
    restored = restored.replace('--max-running-requests ' + str(concurrency),
                                '--max-running-requests 32', 1)
    restored = restored.replace('--cuda-graph-max-bs ' + str(concurrency),
                                '--cuda-graph-max-bs 32', 1)
    require(restored == serve and changed_start.replace(assignment, start_anchor, 1) == start,
            'Private scripts changed unrelated bytes')
    require('bash "$SCRIPT_DIR/serve.sh"' in changed_start,
            'Private start script must use its private serve.sh')
    return {'serve.sh': changed_serve, 'start_server.sh': changed_start}


def make_private_scripts(core, target, concurrency):
    folder = target / 'scripts'
    folder.mkdir(exist_ok=False)
    for name, text in private_script_texts(core.PACKAGE, concurrency).items():
        path = folder / name
        path.write_text(text)
        path.chmod(0o755)
        subprocess.run(['bash', '-n', str(path)], check=True)
    core.write(target / 'private-scripts.json', {
        'package_root': str(core.PACKAGE), 'server_max_running_requests': concurrency,
        'cuda_graph_max_bs': concurrency, 'source_sha256': {
            name: core.sha(core.PACKAGE / 'scripts' / name) for name in ('serve.sh', 'start_server.sh')},
        'private_sha256': {p.name: core.sha(p) for p in sorted(folder.iterdir())},
        'only_changes': ['PACKAGE_DIR remains actual submit root',
                         'max-running-requests 32 -> requested concurrency',
                         'cuda-graph-max-bs 32 -> requested concurrency'],
        'automatic_downgrade': False})


def record_state(core, path, state):
    state['utc'] = core.now()
    core.write(path, state)
    with (path.parent / 'state-events.jsonl').open('a') as stream:
        stream.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + '\n')


def create_container(core, worker, target, owned):
    pair = worker['pair']
    key = pair.replace(',', '')
    name = f'qwen38-{core.RUN}-p{key}'
    exists = subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    require(exists.returncode != 0, 'Refusing to reuse container: ' + name)
    command = ['docker', 'run', '-d', '--name', name, '--label',
               'qwen38.community.run=' + core.RUN, '--label', 'qwen38.community.pair=' + pair,
               '--network', 'host', '--ipc', 'host', '--device', '/dev/kfd', '--device', '/dev/mkfd',
               '--device', '/dev/dri', '--group-add', 'video', '--cap-add', 'SYS_PTRACE',
               '--security-opt', 'seccomp=unconfined', '--security-opt', 'label=disable']
    environment = dict(HIP_VISIBLE_DEVICES=pair, GPU_PAIR=pair, PHYSICAL_PAIR=pair,
                       PORT=str(worker['port']), NCCL_PORT=str(worker['port'] + 1),
                       PATCH_DIR=str(target / 'patch'), MODEL_PATH=str(core.MODEL),
                       DATASET_PATH=str(core.DATASET), VARIANT='candidate',
                       RUN_ID=f'{core.RUN}-p{key}-server', MAX_WAIT='1800', MAX_GATE_WAIT='5',
                       TVM_FFI_CACHE_DIR='/run/qwen38-tvm-ffi', PYTHONDONTWRITEBYTECODE='1',
                       Q38_EXTERNAL_PAIR_LOCK='1')
    for key_, value in environment.items():
        command += ['-e', key_ + '=' + value]
    binds = [(core.PACKAGE, core.PACKAGE, False), (core.CONTROL, core.CONTROL, False),
             (core.MODEL, core.MODEL, True), (core.DATASET, core.DATASET, True),
             (Path('/opt/hyhal'), Path('/opt/hyhal'), True),
             (core.LIBRARY, Path('/run/qwen38-library'), True),
             (core.PACKAGE / 'patch/runtime/tvm-ffi', Path('/run/qwen38-tvm-ffi'), True),
             (core.PACKAGE / 'patch/runtime/direct_register_custom_op.py',
              Path('/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py'), True)]
    for path in (core.PACKAGE / 'config/aiter').glob('*/*.json'):
        binds.append((path, Path('/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs')
                      / path.parent.name / path.name, True))
    for source, destination, readonly in binds:
        require(source.exists(), 'Missing bind source: ' + str(source))
        command += ['-v', f'{source}:{destination}' + (':ro' if readonly else '')]
    command += [IMAGE, 'sleep', 'infinity']
    core.write(target / 'docker-run-command.json', command)
    # Set only after refusing preexisting names; even a partial docker-run failure
    # is cleaned up only after exact label/image/pair validation.
    owned['attempted_name'] = name
    identifier = core.output(command).strip()
    owned['name'] = name
    core.write(target / 'container-created.json', {'name': name, 'id': identifier})
    core.write(target / 'container-inspect.json', core.verify_owned(name, pair))
    with (target / 'pip-freeze.txt').open('x') as stream:
        core.run(['docker', 'exec', name, 'python3', '-I', '-m', 'pip', 'freeze'],
                 stdout=stream, timeout=120)
    return name


def check_server(core, worker, target):
    key = worker['pair'].replace(',', '')
    logdir = core.PACKAGE / 'logs' / f'{core.RUN}-p{key}-server'
    logfile = logdir / 'server.log'
    text = logfile.read_text()
    require(text.count('[Native BF16 AR] Installed') >= 2, 'Native guard installation missing')
    for rank in (0, 1):
        require(f'[Native BF16 AR] Eligible rank={rank} world_size=2' in text,
                f'Native BF16 guard was not exercised on rank {rank}')
    info = json.loads((logdir / 'server-info.json').read_text())
    args = info.get('server_args', info)
    expected = dict(tp_size=2, pp_size=1, context_length=67584,
                    max_running_requests=worker['c'], cuda_graph_max_bs=worker['c'],
                    disable_cuda_graph=False, disable_custom_all_reduce=False,
                    custom_all_reduce_backend='native', attention_backend='fa3',
                    mm_attention_backend='fa3', page_size=64, mem_fraction_static=0.85,
                    max_prefill_tokens=65536, chunked_prefill_size=8192,
                    disable_radix_cache=True, mamba_scheduler_strategy='no_buffer',
                    speculative_num_steps=1, speculative_eagle_topk=1,
                    speculative_num_draft_tokens=2, model_path=str(core.MODEL),
                    served_model_name='Qwen3.8-27B-Channel-INT8-w8a8')
    core.write(target / 'server-info.json', info)
    for key_, value in expected.items():
        require(args.get(key_) == value, f'API parameter mismatch: {key_}: {args.get(key_)!r}')
    require(args.get('speculative_algorithm') in ('NEXTN', 'EAGLE'), 'MTP algorithm differs')
    require(isinstance(args.get('cuda_graph_bs'), list)
            and max(args['cuda_graph_bs']) == worker['c'], 'API graph batch list does not reach requested C')
    core.write(target / 'server-parameter-review.json', {
        'status': 'REQUESTED_SERVER_AND_GRAPH_LIMITS_VERIFIED', 'expected': expected,
        'actual_speculative_algorithm': args['speculative_algorithm'],
        'actual_cuda_graph_bs': args['cuda_graph_bs'], 'native_guard_eligible_ranks': [0, 1],
        'actual_running_request_peak': 'NOT_INFERRED_FROM_CONFIGURATION',
        'automatic_downgrade': False})


class MetricsSampler:
    def __init__(self, core, port, target):
        self.core = core
        self.url = f'http://127.0.0.1:{port}/metrics'
        self.path = target / 'running-metrics.jsonl'
        self.event = threading.Event()
        self.count = 0
        self.error = None
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with self.path.open('x') as stream:
                while not self.event.is_set():
                    sample_started = time.monotonic()
                    row = {'utc': self.core.now(), 'url': self.url,
                           'phase': 'benchmark_including_warmup', 'gauge_aggregation': 'NONE'}
                    try:
                        with opener.open(self.url, timeout=4) as response:
                            row.update(status_code=response.status,
                                       text=response.read().decode('utf-8'))
                    except BaseException as error:
                        row['error'] = repr(error)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    self.count += 1
                    self.event.wait(max(0, 5 - (time.monotonic() - sample_started)))
        except BaseException as error:
            self.error = repr(error)

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
        return {'samples': self.count, 'error': self.error, 'joined': not self.thread.is_alive(),
                'target_start_interval_s': 5, 'http_timeout_s': 4,
                'output': str(self.path), 'gauge_aggregation': 'NONE',
                'requests_during_benchmark': True, 'profiler_or_trace': False}


def snapshot_runtime(core, worker, target, result=None):
    key = worker['pair'].replace(',', '')
    logdir = core.PACKAGE / 'logs' / f'{core.RUN}-p{key}-server'
    if logdir.is_dir():
        destination = target / 'server-final'
        destination.mkdir(exist_ok=True)
        for source in logdir.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)
    if result:
        directory = Path(result['path'])
        core.write(target / 'benchmark-artifact-sha256.json', {
            str(path.relative_to(directory)): {'bytes': path.stat().st_size, 'sha256': core.sha(path)}
            for path in sorted(directory.rglob('*')) if path.is_file()})
    core.write(target / 'runtime-artifact-sha256.json', {
        str(path.relative_to(target)): {'bytes': path.stat().st_size, 'sha256': core.sha(path)}
        for path in sorted(target.rglob('*'))
        if path.is_file() and path.name != 'runtime-artifact-sha256.json'})


def worker_run(worker, settings):
    core = load_core(ROOT, Path(settings['control']), settings['run_id'])
    pair = worker['pair']
    key = pair.replace(',', '')
    statepath = core.CONTROL / f'p{key}/state.json'
    statepath.parent.mkdir(exist_ok=False)
    state = dict(status='PREPARING', pair=pair, client_concurrency=worker['c'],
                 server_max_running_requests=worker['c'], cuda_graph_max_bs=worker['c'],
                 port=worker['port'], pid=os.getpid(), run_id=core.RUN, scenes=[],
                 server_run_id=f'{core.RUN}-p{key}-server',
                 parallel_instances=2, same_host_interference=True, accuracy='NOT_RUN',
                 automatic_downgrade=False)
    record_state(core, statepath, state)
    lease = None
    locked = False
    target = None
    sampler = None
    owned = {}
    result = None
    try:
        lock = Path(f'/data/wangzy/submit1009/result/gpu{key}-owner.lock')
        lock.parent.mkdir(parents=True, exist_ok=True)
        lease = lock.open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        before = core.smi(pair)
        for card, values in before.items():
            require(float(values['HCU use (%)']) == 0
                    and float(values['vram Total Used Memory (MiB)']) <= 16,
                    f'GPU is not idle: {card}: {values}')
        core.write(statepath.parent / 'gpu-before.json', before)
        for port in (worker['port'], worker['port'] + 1, worker['port'] + 2):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
        target = core.make_runtime(pair)
        make_private_scripts(core, target, worker['c'])
        name = create_container(core, worker, target, owned)
        state.update(status='PAIR_GATE', container=name, runtime=str(target))
        record_state(core, statepath, state)
        core.gate(name, pair, target, worker['port'])
        state['status'] = 'MODEL_STARTING'
        record_state(core, statepath, state)
        with (statepath.parent / 'startup.log').open('x') as stream:
            core.run(['docker', 'exec', name, 'bash', str(target / 'scripts/start_server.sh')],
                     stdout=stream, stderr=subprocess.STDOUT, timeout=2000)
        check_server(core, worker, target)
        state['status'] = 'BENCHMARKING'
        record_state(core, statepath, state)
        sampler = MetricsSampler(core, worker['port'], target)
        try:
            sampler.start()
        except BaseException as error:
            sampler.error = repr(error)
        try:
            result = core.bench(name, pair, '4k', worker['c'], 'server' + str(worker['c']))
        finally:
            state['metrics_sampling'] = sampler.stop()
            sampler = None
        state['scenes'].append(result)
        state['status'] = 'BENCHMARK_COMPLETE'
        record_state(core, statepath, state)
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
        record_state(core, statepath, state)
    finally:
        if sampler:
            state['metrics_sampling'] = sampler.stop()
        state['outcome_before_cleanup'] = state['status']
        name = owned.get('name', owned.get('attempted_name'))
        if name:
            try:
                inspected = subprocess.run(['docker', 'inspect', name], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
                if inspected.returncode == 0:
                    core.verify_owned(name, pair)
                    with (statepath.parent / 'container-final-top.txt').open('x') as stream:
                        subprocess.run(['docker', 'top', name, '-eo', 'pid,ppid,args'],
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=30)
                    core.run(['docker', 'stop', '--time', '30', name],
                             stdout=subprocess.DEVNULL, timeout=90)
                    stopped = core.verify_owned(name, pair)
                    require(not stopped['State']['Running'], 'Owned container remains running')
                    core.write(statepath.parent / 'container-stopped.json', stopped)
                state['cleanup_status'] = 'OWNED_CONTAINER_STOPPED_OR_NOT_CREATED'
            except BaseException as error:
                state.update(cleanup_status='FAILED', cleanup_error=repr(error),
                             cleanup_traceback=traceback.format_exc())
                if state['status'] != 'FAILED':
                    state['status'] = 'CLEANUP_FAILED'
        try:
            if locked:
                state['gpu_after'] = core.smi(pair)
        except BaseException as error:
            state['gpu_observation_error'] = repr(error)
        finally:
            if locked:
                fcntl.flock(lease, fcntl.LOCK_UN)
            if lease:
                lease.close()
            state['lock_acquired'] = locked
            state['lock_released'] = True
        try:
            if target:
                snapshot_runtime(core, worker, target, result)
            state['unchanged_default_sources'] = {
                name: core.sha(core.PACKAGE / name) for name in
                ('scripts/serve.sh', 'scripts/start_server.sh', 'patch/sitecustomize.py',
                 'patch/native_bf16_ar_guard.py', 'patch/k100_int8_gemv_unified.so')}
        except BaseException as error:
            state['evidence_collection_error'] = repr(error)
        state['finished_utc'] = core.now()
        record_state(core, statepath, state)
    return state


def prepare(core, settings, ownership):
    require(not core.CONTROL.exists(), 'Control exists; choose a new run ID and control root')
    core.CONTROL.mkdir(parents=True, exist_ok=False)
    ownership['control_created'] = True
    source = Path(__file__).resolve()
    shutil.copy2(source, core.CONTROL / 'controller-source.py')
    core.write(core.CONTROL / 'execution-plan.json', {
        'run_id': core.RUN, 'control': str(core.CONTROL), 'package': str(core.PACKAGE),
        'workers': list(WORKERS), 'automatic_downgrade': False,
        'default_candidate_changed': False, 'metrics_sampling': 'GET /metrics every 5s; raw gauges, no sum',
        'same_host_interference': True, 'settings': settings})
    with (core.CONTROL / 'package-runtime-check.log').open('x') as stream:
        core.run([sys.executable, '-I', str(core.PACKAGE / 'scripts/verify_package.py'), '--runtime'],
                 stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    require(core.output(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).strip() == IMAGE,
            'Declared image is not installed')
    require(core.sha(core.PACKAGE / 'patch/sitecustomize.py') == SITE_SHA, 'Candidate identity mismatch')
    for worker in WORKERS:
        key = worker['pair'].replace(',', '')
        require(not (core.PACKAGE / 'logs' / f'{core.RUN}-p{key}-server').exists(),
                'Server run already exists; choose a new run ID')
        benchmark_id = f"{core.RUN}-p{key}-server{worker['c']}-4k-c{worker['c']}"
        require(not (core.PACKAGE / 'result/runs' / benchmark_id).exists(),
                'Benchmark run already exists; choose a new run ID')
    for port in (31540, 31541, 31542, 31560, 31561, 31562):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', port))
    asset = core.PACKAGE / 'result/evidence/official-alignment-20261017/community-official-20261017-c/model-sha256.json'
    require(asset.is_file(), 'Existing deployed model identity record is missing')
    shutil.copy2(asset, core.CONTROL / 'existing-model-sha256.json')
    core.write(core.CONTROL / 'preparation.json', {
        'status': 'PACKAGE_LIBRARY_AND_IMAGE_READY', 'utc': core.now(),
        'existing_model_manifest': str(asset), 'existing_model_manifest_sha256': core.sha(asset),
        'model_weights_rehashed_this_run': False,
        'source_sha256': {str(path.relative_to(core.PACKAGE)): core.sha(path) for path in
            [core.SOURCE / 'controller.py', core.SOURCE / 'probe_template.py',
             core.PACKAGE / 'scripts/serve.sh', core.PACKAGE / 'scripts/start_server.sh',
             core.PACKAGE / 'scripts/benchmark_reference.sh', core.PACKAGE / 'scripts/benchmark.reference.py']}})


def dry_check(package):
    ast.parse(Path(__file__).read_text())
    core = load_core(package, DEFAULT_CONTROL, DEFAULT_RUN)
    ast.parse((core.SOURCE / 'probe_template.py').read_text())
    require(core.sha(package / 'patch/sitecustomize.py') == SITE_SHA, 'Candidate identity mismatch')
    guard = (package / 'patch/native_bf16_ar_guard.py').read_text()
    require(guard.count("assert os.environ['HIP_VISIBLE_DEVICES'] == '6,7'") == 1,
            'Physical guard adaptation anchor differs')
    for worker in WORKERS:
        for text in private_script_texts(package, worker['c']).values():
            subprocess.run(['bash', '-n'], input=text, text=True, check=True)
    print(json.dumps({'status': 'CPU_DRY_CHECK_PASS', 'workers': list(WORKERS),
                      'gpu_used': False, 'containers_started': False,
                      'files_written': False, 'source_helper_directory': str(core.SOURCE)},
                     ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--dry-check', action='store_true')
    parser.add_argument('--package', type=Path, default=ROOT,
                        help='Package override is accepted for CPU dry-check only')
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    parser.add_argument('--control-root', type=Path, default=DEFAULT_CONTROL)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', args.run_id) is not None,
            'Run ID must contain only A-Za-z0-9_.- and be at most 100 characters')
    if args.dry_check:
        dry_check(args.package.resolve())
        return
    require(args.package == ROOT, 'Actual runs use the fixed declared package root')
    require(args.control_root.is_absolute() and args.control_root.resolve().is_relative_to(Path('/data/wangzy/v2'))
            and args.control_root.resolve() != Path('/data/wangzy/v2'), 'Unsafe control root')
    settings = {'control': str(args.control_root.resolve()), 'run_id': args.run_id}
    core = load_core(ROOT, Path(settings['control']), args.run_id)
    ownership = {}
    try:
        prepare(core, settings, ownership)
        core.write(core.CONTROL / 'controller-state.json', {
            'status': 'RUNNING', 'utc': core.now(), 'pid': os.getpid(), 'workers': list(WORKERS)})
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker_run, worker, settings) for worker in WORKERS]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                core.write(core.CONTROL / 'completed-workers.json', results)
        status = 'COMPLETE' if all(row['status'] == 'BENCHMARK_COMPLETE' for row in results) else 'HAS_FAILURES'
        core.write(core.CONTROL / 'controller-state.json', {
            'status': status, 'utc': core.now(), 'workers': results, 'pid': os.getpid(),
            'official_acceptance': False, 'accuracy': 'NOT_RUN'})
        core.write(core.CONTROL / 'finished.json', {'status': status, 'utc': core.now()})
    except BaseException as error:
        if ownership.get('control_created'):
            core.write(core.CONTROL / 'controller-failure.json', {
                'error': repr(error), 'traceback': traceback.format_exc(), 'utc': core.now()})
            core.write(core.CONTROL / 'controller-state.json', {
                'status': 'FAILED', 'utc': core.now(), 'error': repr(error), 'pid': os.getpid()})
            core.write(core.CONTROL / 'finished.json', {'status': 'FAILED', 'utc': core.now()})
        raise


if __name__ == '__main__':
    main()
