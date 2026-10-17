#!/usr/bin/env python3
"""Guarded native BF16 AllReduce: same-pair unprofiled 4K screening."""
import fcntl
import hashlib
import json
import os
import subprocess
import shutil
import re
import urllib.request
import signal
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/wangzy/submit1009')
C = 'qwen38-submit1009-diag'
IMAGE = 'sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
RUN = 'delivery-native-ar-4k1k-20261010-b'
STATE = ROOT / 'result/delivery-native-ar-4k1k-b-state.json'
OUT = ROOT / 'result' / RUN
F1 = ROOT / 'experiments/native-bf16-ar-patch'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    def deadline(_signum, _frame):
        raise TimeoutError('One-hour delivery deadline guard')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(2700)
    lease = (ROOT / 'result/gpu67-owner.lock').open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), 'Inspect live STATE.pid; never duplicate screening'
    OUT.mkdir(exist_ok=False)
    lease.seek(0); lease.truncate()
    lease.write(json.dumps(dict(pid=os.getpid(), task='Native BF16 copy-in AR separate-item same-pair 4K screening', physical_pair='6,7')))
    lease.flush()

    def state(phase, **kwargs):
        data = dict(phase=phase, pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
                    run_id=RUN, physical_pair='6,7', tp=2, profiler_used=False,
                    observer_enabled=False, default_patch='v1', formal_acceptance=False, **kwargs)
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(STATE)
        print(json.dumps(data), flush=True)

    def docker(command, patch, run_id, benchmark=False):
        env = ['HIP_VISIBLE_DEVICES=6,7', 'PORT=31014', 'NCCL_PORT=31015',
               f'PATCH_DIR={patch}', f'RUN_ID={run_id}', 'DIAG_NO_GRAPH=0', 'MAX_WAIT=600', 'MAX_GATE_WAIT=5']
        if benchmark:
            # Keep two full-size warmups. One measured round is screening only.
            env += ['WARMUP_ROUNDS=2', 'MEASURE_ROUNDS=3', 'MAX_WAIT=600', 'MAX_GATE_WAIT=5']
        cmd = ['docker', 'exec']
        for e in env:
            cmd += ['-e', e]
        subprocess.run(cmd + [C, 'bash', '-lc', command], check=True, timeout=900)

    try:
        obj = json.loads(subprocess.check_output(['docker', 'inspect', C]))[0]
        assert obj['Image'] == IMAGE
        assert 'HIP_VISIBLE_DEVICES=6,7' in obj['Config']['Env']
        assert any(m.get('Source') == str(ROOT) and m.get('Destination') == str(ROOT) for m in obj['Mounts'])
        top = subprocess.check_output(['docker', 'top', C, '-eo', 'pid,ppid,args'], text=True)
        assert not any(s in top for s in ['benchmark.py', 'bench_serving', 'hipprof', 'sglang', 'test_route.py', 'probe_prepack.py'])
        policy = json.loads((ROOT / 'manifest/acceptance-policy-20pct.json').read_text())
        assert policy['status'] == 'ACTIVE'
        assert not (ROOT / 'result/final-summary.json').exists()
        assert sha(ROOT / 'patch/sitecustomize.py') == 'f1789532d58daa66187ba901d3d7513ff142cb33fafc7793ce44f4e821464495'
        gate_path = ROOT / 'result/native-bf16-ar-numerical-20261010-b/gate.json'
        gate = json.loads(gate_path.read_text())
        assert gate['status'] == 'NUMERICAL_GRAPH_PASS'
        assert gate['candidate_site_sha256'] == sha(F1/'sitecustomize.py')
        assert all(r['cases']==9 and r['excluded_routes']==10 and r['full_nvlink'] is False for r in gate['ranks'])
        assert not list(F1.glob('start*.pending'))
        prior = json.loads((ROOT/'result/native-bf16-ar-numerical-state.json').read_text())
        assert prior['phase'] == 'native-bf16-ar-numerical-complete'
        assert not (Path('/proc')/str(prior['pid'])).exists()
        assert (ROOT/'result/native-bf16-ar-numerical-20261010-b/finished').exists()
        assert (ROOT/'result/native-bf16-ar-numerical-20261010-b/complete-sha256.json').exists()
        assert sha(F1/'sitecustomize.py') == 'da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610'
        deployment = ROOT/'experiments/tp-communication/native-bf16'
        assert sha(deployment/'serve_native_bf16_ar.sh') == '317a695ba8a3d78d2f25401ea2b5a94a51c7b7ef57b42e4c1af2564b2d00f97d'
        assert sha(deployment/'start_native_bf16_ar.sh') == 'c7d198af5105e904ae8d01e6daa2649290e39361d0052120bb919d6dcb398945'
        ab_obj=json.loads(subprocess.check_output(['docker','inspect','qwen38-submit1009-ab']))[0]
        ab_top=subprocess.check_output(['docker','top','qwen38-submit1009-ab','-eo','pid,ppid,args'],text=True) if ab_obj['State']['Running'] else 'Stopped by user delivery request'
        assert not any(s in ab_top for s in ['benchmark.py','bench_serving','hipprof','test_route.py'])

        f1_manifest = json.loads((F1 / 'native-ar-manifest.json').read_text())
        for name, digest in f1_manifest['files'].items():
            assert sha(F1 / name) == digest, name
        tile_manifest = json.loads((ROOT / 'result/tile-route-candidate-manifest.json').read_text())
        for name, digest in tile_manifest.items():
            assert sha(ROOT / 'experiments/tile-route-patch' / name) == digest, name
        save(OUT / 'preflight.json', dict(container=C, image=obj['Image'], processes=top,
             numerical_gate=gate, numerical_gate_sha256=sha(gate_path), f1_manifest=f1_manifest,
             acceptance_policy=policy, overlap='4,5 owned AB stopped; no other task GPU experiments'))
        cases = [('v1', ROOT / 'patch'), ('native-bf16-ar', F1), ('v1-return', ROOT / 'patch')]
        summaries = {}
        for kind, patch in cases:
            run_id = RUN + '-' + kind
            target = ROOT / 'result' / run_id
            target.mkdir(exist_ok=False)
            save(target / 'screen-config.json', dict(kind=kind, physical_pair='6,7', tp=2,
                 input_len=4096, output_len=1024, concurrency=8, seed=20260912,
                 warmup_rounds=2, measure_rounds=3, requests_per_round=16,
                 site_sha256=sha(patch / 'sitecustomize.py'), profiler_used=False,
                 performance_acceptance=False, policy='each 4K/64K/C32 >=20%; matched >=3 rounds required'))
            state('starting-screen-service', kind=kind, current_run=run_id)
            subprocess.run(['docker', 'restart', C], check=True, timeout=90)
            start_script = (deployment/'start_native_bf16_ar.sh') if kind == 'native-bf16-ar' else ROOT/'scripts/start_server.sh'
            docker(f'bash {start_script}', patch, run_id + '-server')
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            info = json.loads(opener.open('http://127.0.0.1:31014/get_server_info', timeout=20).read())
            save(target / 'server-info.json', info)
            log = ROOT / 'logs' / (run_id + '-server') / 'server.log'
            if kind == 'native-bf16-ar':
                text = log.read_text()
                marker = '[Native BF16 AR] Installed BF16/TP2/copy-in guard; unsupported tensors retain RCCL'
                assert text.count(marker) >= 2, 'Require candidate guard import in both TP ranks'
                assert info['custom_all_reduce_backend'] == 'native' and info['disable_custom_all_reduce'] is False
                assert info['tp_size'] == 2 and info['cuda_graph_max_bs'] == 32
                assert info['max_running_requests'] == 32 and info['speculative_num_steps'] == 1
                enabled = re.findall(r'\[AR\] custom_all_reduce_backend=native[^\n]*disabled=False[^\n]*world_size=2', text)
                assert len(enabled) >= 2, 'No silent RCCL fallback of entire communicator'
                for rank in (0,1):
                    assert f'[Native BF16 AR] Eligible rank={rank} world_size=2' in text
                save(target / 'patch-load-review.json', dict(status='SERVICE_GUARD_AND_BACKEND_LOADED_REVIEW_PENDING',
                    actual_model_profile_graph_gate='REVIEW_REQUIRED', formal_operator_selection=False,
                    native_enabled_communicators=len(enabled), installed_marker_count=text.count(marker),
                    warning='First activation/loading logs do not prove actual Graph replay association'))
            else:
                assert info['disable_custom_all_reduce'] is True and info['custom_all_reduce_backend'] == 'off'
            state('benchmarking-screen-4k', kind=kind, current_run=run_id)
            docker(f'bash {ROOT}/scripts/benchmark.sh 4k', patch, run_id, benchmark=True)
            summary = json.loads((target / '4k/validated_summary.json').read_text())
            assert summary['status'] == 'PASS' and summary['measured_rounds'] == 3 and summary['requests'] == 48
            summaries[kind] = summary
            shutil.copy2(log,target/'server.log')
            save(target / 'raw-sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in target.rglob('*') if p.is_file()})
        base = (summaries['v1']['mean_output_tps'] + summaries['v1-return']['mean_output_tps']) / 2
        fusion = summaries['native-bf16-ar']['mean_output_tps']
        tails = {name: {key: (s[key] / summaries['v1'][key] - 1) * 100
                 for key in ['ttft_s_p95_mean_of_rounds', 'ttft_s_p99_mean_of_rounds',
                             'tpot_ms_p95_mean_of_rounds', 'tpot_ms_p99_mean_of_rounds']}
                 for name, s in summaries.items()}
        comparison = dict(status='THREE_ROUND_4K_PERFORMANCE_ONLY', formal_acceptance=False,
             candidate_site_sha256=sha(F1 / 'sitecustomize.py'), physical_pair='6,7', tp=2, formal_operator_selection=False,
             actual_model_graph_gate='REVIEW_REQUIRED',
             item='BF16 copy-in-only native one-stage reduction; other routes retain RCCL',
             summaries=summaries, combined_v1_tps=base, native_ar_gain_vs_v1_percent=(fusion/base-1)*100,
             v1_return_drift_percent=(summaries['v1-return']['mean_output_tps']/summaries['v1']['mean_output_tps']-1)*100,
             tails_vs_initial_v1_percent=tails, rounds_per_variant=3, target_gain_percent=20,
             target_reached_in_screen=fusion/base >= 1.20,
             next='Timed 4K/1K delivery only; 64K/C32, actual-model Graph, clean-room and GSM8K incomplete. No FROZEN_PASS.')
        save(OUT / 'comparison.json', comparison)
        save(OUT/'deployment-sha256.json', {p.name:sha(p) for p in deployment.glob('*.sh')})
        (OUT / 'finished').write_text(datetime.now(timezone.utc).isoformat() + '\n')
        state('delivery-4k1k-complete', comparison=str((OUT / 'comparison.json').relative_to(ROOT)),
              native_ar_gain_vs_v1_percent=comparison['native_ar_gain_vs_v1_percent'],
              rounds_per_variant=3)
    except Exception as exc:
        state('failed', error=repr(exc))
        if 'obj' in locals() and obj['Image'] == IMAGE:
            subprocess.run(['docker', 'stop', '--time', '20', C], check=True, timeout=120)
        raise
    finally:
        signal.alarm(0)
        subprocess.run(['docker', 'stop', '--time', '20', C], check=True, timeout=120)
        fcntl.flock(lease, fcntl.LOCK_UN); lease.close()


if __name__ == '__main__':
    main()
