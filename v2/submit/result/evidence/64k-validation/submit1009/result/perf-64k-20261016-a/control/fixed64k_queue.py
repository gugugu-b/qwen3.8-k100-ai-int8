#!/usr/bin/env python3
"""Same-pair 64K/1K performance validation; no quality evaluation or promotion."""
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("TASK_ROOT", "/data/wangzy/submit1009"))
RUNTIME = ROOT / "experiments/performance-validation-20261016"
RUN = "perf-64k-20261016-a"
PAIR = "4,5"
CONTAINER = "qwen38-submit1009-perf45-20261016"
PORT = 31024
IMAGE = "sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
OUT = ROOT / "result" / RUN
STATE = ROOT / "result" / (RUN + "-state.json")
LOG_ROOT = Path(os.environ.get("SERVICE_LOG_ROOT", str(RUNTIME / "logs")))
BASE_SHA = "f1789532d58daa66187ba901d3d7513ff142cb33fafc7793ce44f4e821464495"
SITE_SHA = "da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    assert not path.exists(), "Refusing overwrite: " + str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def state(phase, **extra):
    data = dict(phase=phase, pid=os.getpid(), utc=stamp(), run_id=RUN,
                physical_pair=PAIR, tp=2, scene="64k1k-c8", quality_evaluation=False,
                formal_acceptance=False, frozen=False, **extra)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATE)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def checked(command, timeout=120, log=None):
    if log:
        with Path(log).open("x") as handle:
            return subprocess.run(command, check=True, timeout=timeout,
                                  stdout=handle, stderr=subprocess.STDOUT)
    return subprocess.run(command, check=True, timeout=timeout)


def inspect_owned():
    obj = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER]))[0]
    assert obj["Image"] == IMAGE
    assert "HIP_VISIBLE_DEVICES=" + PAIR in obj["Config"]["Env"]
    assert any(m.get("Source") == str(ROOT) and m.get("Destination") == str(ROOT)
               for m in obj["Mounts"]), "Task root must be mounted unchanged"
    return obj


def dexec(command, variant, patch, server_run=None, timeout=900, log=None):
    env = {"HIP_VISIBLE_DEVICES": PAIR, "PHYSICAL_PAIR": PAIR,
           "PORT": str(PORT), "NCCL_PORT": "31025", "VARIANT": variant,
           "PATCH_DIR": str(patch), "MAX_WAIT": "900", "MAX_GATE_WAIT": "5"}
    if server_run:
        env["RUN_ID"] = server_run
    cmd = ["docker", "exec"]
    for key, value in env.items():
        cmd += ["-e", key + "=" + value]
    checked(cmd + [CONTAINER] + command, timeout=timeout, log=log)


def server_info():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    assert opener.open("http://127.0.0.1:%d/health" % PORT, timeout=20).status == 200
    return json.loads(opener.open("http://127.0.0.1:%d/get_server_info" % PORT,
                                 timeout=30).read())


def validate_api(info, candidate):
    expected = {"tp_size": 2, "pp_size": 1, "context_length": 67584,
                "mem_fraction_static": 0.85, "max_running_requests": 32,
                "max_prefill_tokens": 65536, "chunked_prefill_size": 8192,
                "mamba_scheduler_strategy": "no_buffer", "cuda_graph_max_bs": 32,
                "disable_cuda_graph": False, "disable_radix_cache": True,
                "attention_backend": "fa3", "mm_attention_backend": "fa3",
                "page_size": 64, "speculative_num_steps": 1,
                "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 2}
    for key, value in expected.items():
        assert info.get(key) == value, (key, info.get(key), value)
    assert info["speculative_algorithm"] in ("NEXTN", "EAGLE")
    assert info["custom_all_reduce_backend"] == ("native" if candidate else "off")
    assert info["disable_custom_all_reduce"] is (not candidate)


def validate_requests(directory):
    for phase, count, requests in (("warmup", 1, 8), ("round", 3, 16)):
        rounds = sorted(directory.glob(phase + "-*/requests.jsonl"))
        assert len(rounds) == count, (phase, len(rounds))
        for path in rounds:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            assert len(rows) == requests
            assert len({r["request_index"] for r in rows}) == requests
            for row in rows:
                assert row["status_code"] == 200 and row["done"] is True
                assert not row.get("error") and not row.get("replacement_chars", 0)
                assert row["prompt_tokens"] == 65536 and row["completion_tokens"] == 1024
                for metric in ("e2e_s", "ttft_s", "tpot_ms"):
                    assert math.isfinite(row[metric]) and row[metric] > 0
            summary = json.loads((path.parent / "summary.json").read_text())
            assert summary["status"] == "PASS" and summary["successful"] == requests


def compare(summaries):
    initial, candidate, returned = (summaries[k] for k in ("v1", "candidate", "v1-return"))
    base = statistics.mean([initial["mean_output_tps"], returned["mean_output_tps"]])
    gain = (candidate["mean_output_tps"] / base - 1) * 100
    drift = (returned["mean_output_tps"] / initial["mean_output_tps"] - 1) * 100
    variation = max(abs(drift), *(s["spread_percent"] for s in summaries.values()))
    paired = [(b / statistics.mean([a, c]) - 1) * 100 for a, b, c in zip(
        initial["output_tps_rounds"], candidate["output_tps_rounds"], returned["output_tps_rounds"])]
    tails = {}
    round_tails = {}
    for metric in ("ttft_s", "tpot_ms", "e2e_s"):
        for quantile in ("p95", "p99"):
            name = metric + "_" + quantile
            vals = {}
            for kind in summaries:
                vals[kind] = [json.loads(p.read_text())[metric][quantile]
                              for p in sorted((OUT / "variants" / kind / "64k").glob("round-*/summary.json"))]
            denominator = statistics.mean(vals["v1"] + vals["v1-return"])
            tails[name] = (statistics.mean(vals["candidate"]) / denominator - 1) * 100
            round_tails[name] = [(b / statistics.mean([a, c]) - 1) * 100
                                for a, b, c in zip(vals["v1"], vals["candidate"], vals["v1-return"])]
    return dict(status="THREE_ROUND_64K1K_PERFORMANCE_VALIDATION_COMPLETE", physical_pair=PAIR,
                tp=2, summaries=summaries, combined_v1_tps=base, gain_percent=gain,
                paired_round_gain_percent=paired, v1_return_drift_percent=drift,
                observed_variation_reference_percent=variation,
                variation_note="Input differs by round, matches across variants; not a statistical confidence bound",
                gain_exceeds_observed_variation=gain > variation,
                mean_tail_changes_percent=tails, paired_round_tail_changes_percent=round_tails,
                mean_tail_5pct_pass=all(v <= 5 for v in tails.values()),
                paired_round_tail_5pct_pass=all(v <= 5 for vs in round_tails.values() for v in vs),
                measured_20pct_pass=gain >= 20, formal_acceptance=False, frozen=False,
                actual_model_graph_gate="REVIEW_REQUIRED", quality_evaluation="NOT_RUN_USER_REQUEST")


def main():
    def interrupted(signum, _frame):
        raise TimeoutError("Queue interrupted/deadline signal " + str(signum))
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    lease = (ROOT / "result/gpu45-owner.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists() and not OUT.exists(), "Inspect prior state/PID; never restart an existing RUN"
    OUT.mkdir()
    lease.seek(0); lease.truncate()
    lease.write(json.dumps(dict(pid=os.getpid(), task=RUN, physical_pair=PAIR, utc=stamp())) + "\n")
    lease.flush()
    signal.alarm(3 * 60 * 60)
    owned = False
    success = False
    active_server_run = None
    failure = None
    try:
        obj = inspect_owned(); owned = True
        assert obj["State"]["Running"]
        top = subprocess.check_output(["docker", "top", CONTAINER, "-eo", "pid,ppid,args"], text=True)
        assert not any(s in top for s in ("sglang", "benchmark.py", "bench_serving", "hipprof", "probe_"))
        gate_path = RUNTIME / "candidate45-gate.json"
        gate = json.loads(gate_path.read_text())
        assert gate["status"] == "NUMERICAL_GRAPH_PASS"
        assert gate.get("physical_pair") == PAIR, "Require this physical pair's numerical/Graph gate"
        baseline, patch = RUNTIME / "baseline-v1", RUNTIME / "candidate45"
        assert sha(baseline / "sitecustomize.py") == BASE_SHA
        assert sha(patch / "sitecustomize.py") == SITE_SHA
        assert gate["candidate_site_sha256"] == SITE_SHA
        assert gate["candidate_guard_sha256"] == sha(patch / "native_bf16_ar_guard.py")
        assert not list(patch.glob("start*.pending"))
        control = OUT / "control"; control.mkdir()
        shutil.copy2(Path(__file__), control / "fixed64k_queue.py")
        shutil.copy2(gate_path, control / gate_path.name)
        for name in ("start_service.sh", "serve.sh", "benchmark.py", "validate_results.py"):
            shutil.copy2(RUNTIME / "scripts" / name, control / name)
        for name, src in (("candidate45", patch), ("baseline-v1", baseline)):
            shutil.copytree(src, control / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        save(OUT / "preflight.json", dict(utc=stamp(), container=CONTAINER, image=obj["Image"],
             container_mounts=obj["Mounts"], processes=top, candidate_gate=gate, gate_sha256=sha(gate_path),
             source_sha256=sha(Path(__file__)), profiler_used=False, quality_evaluation=False,
             overlap_note=os.environ.get("OVERLAP_NOTE", "6,7 independent ShareGPT queue may overlap; record live UTC spans")))
        summaries = {}
        for kind, variant, source in (("v1", "baseline-v1", baseline), ("candidate", "candidate", patch),
                                       ("v1-return", "baseline-v1", baseline)):
            target = OUT / "variants" / kind
            target.mkdir(parents=True)
            active_server_run = RUN + "-" + kind + "-server"
            save(target / "run-config.json", dict(kind=kind, variant=variant, physical_pair=PAIR, tp=2,
                 input_len=65536, output_len=1024, concurrency=8, seed=20260912,
                 warmup_rounds=1, warmup_requests=8, measure_rounds=3, requests_per_round=16,
                 site_sha256=sha(source / "sitecustomize.py"), begin_utc=stamp(), profiler_used=False))
            state("starting-service", kind=kind, server_run=active_server_run)
            inspect_owned()
            checked(["docker", "restart", CONTAINER], timeout=120)
            dexec(["bash", str(RUNTIME / "scripts/start_service.sh")], variant, source,
                  active_server_run, timeout=1100, log=target / "startup.log")
            info = server_info(); validate_api(info, kind == "candidate")
            save(target / "server-info.json", info)
            live = LOG_ROOT / active_server_run
            assert (live / "server.log").is_file(), "Check SERVICE_LOG_ROOT/start_service.sh log interface"
            if kind == "candidate":
                text = (live / "server.log").read_text(errors="replace")
                enabled = re.findall(r"\[AR\] custom_all_reduce_backend=native[^\n]*disabled=False[^\n]*world_size=2", text)
                assert len(enabled) >= 2, "Both native communicators must be enabled"
                assert text.count("[Native BF16 AR] Installed BF16/TP2/copy-in guard") >= 2
                for rank in (0, 1):
                    assert "[Native BF16 AR] Eligible rank=%d world_size=2" % rank in text
                save(target / "candidate-load-review.json", dict(status="BOTH_TP_RANKS_NATIVE_ENABLED_ELIGIBLE",
                     actual_model_graph_gate="REVIEW_REQUIRED", enabled_communicators=len(enabled)))
            state("benchmarking-64k1k", kind=kind, begin_utc=stamp())
            dexec(["python3", "-u", str(RUNTIME / "scripts/benchmark.py"),
                   "--base-url", "http://127.0.0.1:%d" % PORT, "--model", "Qwen3.8-27B-Channel-INT8-w8a8",
                   "--output-dir", str(target / "64k"), "--input-len", "65536", "--output-len", "1024",
                   "--concurrency", "8", "--seed", "20260912", "--warmup-rounds", "1",
                   "--measure-rounds", "3", "--requests-per-round", "16", "--timeout", "7200"],
                  variant, source, RUN + "-" + kind, timeout=7200, log=target / "bench.log")
            validate_requests(target / "64k")
            dexec(["python3", str(RUNTIME / "scripts/validate_results.py"), str(target / "64k"),
                   "--rounds", "3", "--input-len", "65536"], variant, source,
                  timeout=120, log=target / "validation.log")
            summary = json.loads((target / "64k/validated_summary.json").read_text())
            assert summary["status"] == "PASS" and summary["requests"] == 48
            summaries[kind] = summary
            for name in ("server.log", "server.pid"):
                assert (live / name).is_file(); shutil.copy2(live / name, target / name)
            save(target / "completed.json", dict(utc=stamp(), requests=48, warmup_requests=8,
                                                  status="REQUEST_INTEGRITY_PASS"))
        save(OUT / "comparison.json", compare(summaries))
        success = True
    except BaseException as exc:
        failure = repr(exc)
        save(OUT / "failure.json", dict(utc=stamp(), error=failure, traceback=traceback.format_exc(),
                                        active_server_run=active_server_run))
        state("failed", error=failure, active_server_run=active_server_run)
    finally:
        signal.alarm(0)
        cleanup = dict(utc=stamp(), container=CONTAINER, owned_verified=owned)
        if owned:
            try:
                inspect_owned()
                result = subprocess.run(["docker", "stop", "--time", "20", CONTAINER], timeout=120,
                                        capture_output=True, text=True)
                cleanup.update(stop_returncode=result.returncode, stop_stdout=result.stdout,
                               stop_stderr=result.stderr, running=inspect_owned()["State"]["Running"])
                assert result.returncode == 0 and cleanup["running"] is False
            except BaseException as exc:
                cleanup["error"] = repr(exc); success = False
        if active_server_run:
            live = LOG_ROOT / active_server_run
            for name in ("server.log", "server.pid"):
                if (live / name).is_file():
                    shutil.copy2(live / name, OUT / ("last-" + name))
        save(OUT / "cleanup.json", cleanup)
        state("performance-validation-complete" if success else "failed", error=failure,
              all_variants_completed=success, own_container_stopped=cleanup.get("running") is False)
        shutil.copy2(STATE, OUT / "state-final.json")
        if success:
            (OUT / "finished").write_text(stamp() + "\n")
        save(OUT / "complete-sha256.json", {str(p.relative_to(ROOT)): sha(p)
             for p in sorted(OUT.rglob("*")) if p.is_file()})
        fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
