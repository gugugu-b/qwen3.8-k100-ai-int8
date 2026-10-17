#!/usr/bin/env python3
"""Freeze completed delivery evidence and prove task card release."""
import fcntl
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/wangzy/submit1009')
RUN = 'delivery-native-ar-4k1k-20261010-b'
OUT = ROOT / 'result' / RUN
state = json.loads((ROOT / 'result/delivery-native-ar-4k1k-b-state.json').read_text())
assert state['phase'] == 'delivery-4k1k-complete' and (OUT / 'finished').is_file()
assert not (Path('/proc') / str(state['pid'])).exists(), 'Wait for queue finally cleanup and exit'
assert not (OUT / 'complete-sha256.json').exists(), 'Immutable archive already exists'
leases = []
for pair in ('45', '67'):
    f = (ROOT / ('result/gpu' + pair + '-owner.lock')).open('a+')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    leases.append(f)
containers = {}
for name in ('qwen38-submit1009-ab', 'qwen38-submit1009-diag', 'qwen38-submit1009-final', 'Qwen3.8-27B-sglang0.5.12-tuning1'):
    p = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if p.returncode:
        containers[name] = dict(exists=False)
        continue
    x = json.loads(p.stdout)[0]
    assert x['State']['Running'] is False, (name, 'Task container still running')
    containers[name] = dict(exists=True, running=False, pid=x['State']['Pid'], image=x['Image'],
                            finished_at=x['State']['FinishedAt'], intentional_stop=True)
control = OUT / 'archive-evidence'
control.mkdir(exist_ok=False)
gpu = subprocess.run(['/opt/dtk/bin/rocm-smi', '-d', '4', '5', '6', '7', '--showuse', '--showmeminfo', 'vram', '--json'],
                     capture_output=True, text=True, timeout=30)
release = dict(utc=datetime.now(timezone.utc).isoformat(), status='ALL_TASK_CONTAINERS_STOPPED_LOCKS_FREE',
 physical_devices=[4,5,6,7], containers=containers, host_queue_pid=state['pid'], host_queue_alive=False,
 gpu45_lock='FREE', gpu67_lock='FREE', gpu_memory_probe_returncode=gpu.returncode,
 gpu_memory_probe_stdout=gpu.stdout, gpu_memory_probe_stderr=gpu.stderr,
 reason='Human timed delivery and release request; no final serving endpoint remains active')
(control / 'card-release.json').write_text(json.dumps(release, indent=2) + '\n')
if gpu.returncode != 0:
    print('Device memory utility unavailable; container/PID and lock release evidence retained')
for f in ['result/delivery-native-ar-4k1k-b-state.json', 'result/' + RUN + '-launch.json',
          'result/delivery-native-ar-4k1k-state.json', 'result/delivery-native-ar-4k1k-20261010-a-launch.json',
          'result/delivery-native-ar-4k1k-20261010-a-gpu45-release.json',
          'logs/' + RUN + '-job.log', 'logs/delivery-native-ar-4k1k-20261010-a-job.log',
          'manifest/acceptance-policy-20pct.json', 'manifest/artifact-storage-policy.json',
          '02-测试与结果/delivery-4k-20261010-work/run_delivery_4k1k.py',
          '02-测试与结果/delivery-4k-20261010-work/run_delivery_4k1k_b.py',
          '02-测试与结果/delivery-4k-20261010-work/review_delivery_4k.py',
          '02-测试与结果/delivery-4k-20261010-work/archive_delivery.py']:
    src = ROOT / f
    dst = control / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
shutil.copy2('/data/wangzy/bench/benchmark.py', control / 'benchmark.py')
for kind in ('v1', 'native-bf16-ar', 'v1-return'):
    src = ROOT / 'logs' / (RUN + '-' + kind + '-server')
    dst = control / 'server-logs' / kind
    dst.mkdir(parents=True)
    for name in ('server.log', 'server.pid'):
        shutil.copy2(src / name, dst / name)
paths = [ROOT / 'scripts/validate_results.py']
for folder in [OUT] + [ROOT / 'result' / (RUN + '-' + k) for k in ('v1', 'native-bf16-ar', 'v1-return')]:
    paths.extend(p for p in folder.rglob('*') if p.is_file())
manifest = {}
for path in sorted(paths):
    assert path.stat().st_size < 16 * 1024 * 1024, ('Do not download large artifacts', path)
    manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'complete-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
for f in leases:
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()
print(json.dumps(dict(status='DELIVERY_ARCHIVE_FROZEN', files=len(manifest), card_release=release)))
