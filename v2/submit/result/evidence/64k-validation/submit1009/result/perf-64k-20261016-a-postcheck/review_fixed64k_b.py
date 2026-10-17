#!/usr/bin/env python3
"""CPU-only independent review of a completed, immutable 64K/1K validation RUN."""
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = int(math.floor(position)), int(math.ceil(position))
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    values = [float(value) for value in values if value is not None]
    assert values and all(math.isfinite(value) for value in values)
    return dict(count=len(values), mean=sum(values) / len(values), min=min(values), max=max(values),
                p50=percentile(values, 50), p90=percentile(values, 90),
                p95=percentile(values, 95), p99=percentile(values, 99))


def benchmark_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode()


def saved_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def rebuild_round(directory, phase, index, requested):
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == requested, directory
    assert sorted(row["request_index"] for row in rows) == list(range(requested)), directory
    rows.sort(key=lambda row: row["request_index"])
    for row in rows:
        assert row["status_code"] == 200 and row["done"] is True and row["error"] is None, directory
        assert row["prompt_tokens"] == 65536 and row["completion_tokens"] == 1024, directory
        assert row["replacement_chars"] == 0, directory
        assert row["total_tokens"] == 66560, directory
        for metric in ("e2e_s", "ttft_s", "tpot_ms"):
            assert math.isfinite(row[metric]) and row[metric] > 0, (directory, metric)
        assert row["e2e_s"] > row["ttft_s"]
        expected_tpot = 1000.0 * (row["e2e_s"] - row["ttft_s"]) / 1023.0
        assert expected_tpot == row["tpot_ms"], directory
    original = json.loads((directory / "summary.json").read_text())
    wall = original["wall_time_s"]
    assert math.isfinite(wall) and wall > 0
    assert wall >= max(row["e2e_s"] for row in rows), directory
    rebuilt = dict(status="PASS", phase=phase, round=index, input_len=65536, output_len=1024,
                   concurrency=8, requested=requested, successful=requested, exact_length=requested,
                   wall_time_s=wall, request_throughput_rps=requested / wall,
                   input_throughput_tps=sum(row["prompt_tokens"] for row in rows) / wall,
                   output_throughput_tps=sum(row["completion_tokens"] for row in rows) / wall,
                   total_throughput_tps=sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows) / wall,
                   e2e_s=distribution([row["e2e_s"] for row in rows]),
                   ttft_s=distribution([row["ttft_s"] for row in rows]),
                   tpot_ms=distribution([row["tpot_ms"] for row in rows]), replacement_chars=0)
    assert benchmark_json(rebuilt) == (directory / "summary.json").read_bytes(), directory
    return rebuilt


def rebuild_validated(rounds):
    tps = [summary["output_throughput_tps"] for summary in rounds]
    report = dict(status="PASS", measured_rounds=3, requests=48, output_tps_rounds=tps,
                  mean_output_tps=statistics.mean(tps),
                  spread_percent=(max(tps) - min(tps)) / statistics.mean(tps) * 100,
                  mean_wall_s=statistics.mean(summary["wall_time_s"] for summary in rounds))
    for metric in ("ttft_s", "tpot_ms"):
        for quantile in ("p50", "p95", "p99"):
            values = [summary[metric][quantile] for summary in rounds]
            report[metric + "_" + quantile + "_mean_of_rounds"] = statistics.mean(values)
            report[metric + "_" + quantile + "_range"] = [min(values), max(values)]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Task root or small-file mirror containing result/RUN")
    parser.add_argument("--out", type=Path, required=True, help="New review directory; existing paths are refused")
    parser.add_argument("--run", default="perf-64k-20261016-a")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    control = root / "result" / args.run
    assert control.is_dir() and (control / "finished").is_file()
    assert not output.exists(), "Refusing to overwrite any review output"
    assert output != control and control not in output.parents, "Review output must be outside immutable RUN"
    output.mkdir(parents=True)
    manifest_path = control / "complete-sha256.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest and isinstance(manifest, dict)
    for rel, digest in manifest.items():
        path = root / rel
        assert not Path(rel).is_absolute() and path.resolve().is_relative_to(root), rel
        assert path.is_file() and sha(path) == digest, rel
    actual = {str(path.relative_to(root)) for path in control.rglob("*") if path.is_file() and path != manifest_path}
    assert actual == set(manifest), "Manifest must cover every immutable RUN file exactly"
    final = json.loads((control / "state-final.json").read_text())
    assert final["phase"] == "performance-validation-complete" and final["all_variants_completed"] is True
    assert final["own_container_stopped"] is True and final["physical_pair"] == "4,5" and final["tp"] == 2
    cleanup = json.loads((control / "cleanup.json").read_text())
    assert cleanup["running"] is False and cleanup["stop_returncode"] == 0
    gate = json.loads((control / "control/candidate45-gate.json").read_text())
    assert gate["status"] == "NUMERICAL_GRAPH_PASS" and gate["physical_pair"] == "4,5"
    assert gate["candidate_guard_sha256"] == sha(control / "control/candidate45/native_bf16_ar_guard.py")
    assert gate["candidate_site_sha256"] == sha(control / "control/candidate45/sitecustomize.py")
    summaries, infos, configs, measured, round_sha = {}, {}, {}, {}, {}
    for kind in ("v1", "candidate", "v1-return"):
        folder, scene = control / "variants" / kind, control / "variants" / kind / "64k"
        configs[kind] = json.loads((scene / "config.json").read_text())
        expected = dict(input_len=65536, output_len=1024, concurrency=8, seed=20260912,
                        warmup_rounds=1, measure_rounds=3, requests_per_round=16, timeout=7200,
                        model="Qwen3.8-27B-Channel-INT8-w8a8", base_url="http://127.0.0.1:31024")
        for key, value in expected.items():
            assert configs[kind][key] == value, (kind, key)
        rounds = []
        round_sha[kind] = {}
        for prefix, phase, count, n in (("warmup", "warmup", 1, 8), ("round", "measure", 3, 16)):
            dirs = sorted(path for path in scene.glob(prefix + "-*") if path.is_dir())
            assert [path.name for path in dirs] == [prefix + "-%02d" % i for i in range(1, count + 1)]
            for index, directory in enumerate(dirs, 1):
                rebuilt = rebuild_round(directory, phase, index, n)
                round_sha[kind][directory.name] = sha(directory / "summary.json")
                if phase == "measure":
                    rounds.append(rebuilt)
        measured[kind] = rounds
        validated = rebuild_validated(rounds)
        assert saved_json(validated) == (scene / "validated_summary.json").read_bytes(), kind
        summaries[kind] = validated
        aggregate = dict(status="PASS", base_url=configs[kind]["base_url"], model=configs[kind]["model"],
                         input_len=65536, output_len=1024, concurrency=8, warmup_rounds=1,
                         measure_rounds=3, requests_per_round=16, successful_rounds=3,
                         output_throughput_tps=distribution([s["output_throughput_tps"] for s in rounds]),
                         ttft_p50_s=distribution([s["ttft_s"]["p50"] for s in rounds]),
                         ttft_p95_s=distribution([s["ttft_s"]["p95"] for s in rounds]),
                         tpot_p50_ms=distribution([s["tpot_ms"]["p50"] for s in rounds]),
                         tpot_p95_ms=distribution([s["tpot_ms"]["p95"] for s in rounds]), rounds=rounds,
                         method="fixed random token IDs; two warmup waves; three measured rounds")
        assert benchmark_json(aggregate) == (scene / "summary.json").read_bytes(), kind
        copied = output / kind
        copied.mkdir()
        for directory in sorted(scene.glob("round-*")):
            shutil.copytree(directory, copied / directory.name)
        with (output / (kind + "-validation.log")).open("x") as handle:
            subprocess.run([sys.executable, str(control / "control/validate_results.py"), str(copied),
                            "--rounds", "3", "--input-len", "65536"], check=True,
                           stdout=handle, stderr=subprocess.STDOUT)
        assert (copied / "validated_summary.json").read_bytes() == (scene / "validated_summary.json").read_bytes()
        infos[kind] = json.loads((folder / "server-info.json").read_text())
        assert infos[kind]["custom_all_reduce_backend"] == ("native" if kind == "candidate" else "off")
        assert infos[kind]["disable_custom_all_reduce"] is (kind != "candidate")
    for key in configs["v1"]:
        if key != "output_dir":
            assert len({json.dumps(c[key], sort_keys=True) for c in configs.values()}) == 1, key
    same_api = ["tp_size", "pp_size", "attention_backend", "mm_attention_backend", "page_size",
                "context_length", "disable_radix_cache", "mem_fraction_static", "max_running_requests",
                "max_prefill_tokens", "chunked_prefill_size", "mamba_scheduler_strategy", "cuda_graph_max_bs",
                "disable_cuda_graph", "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
                "speculative_num_draft_tokens", "linear_attn_backend", "mamba_backend"]
    for key in same_api:
        assert key in infos["v1"]
        assert len({json.dumps(info[key], sort_keys=True) for info in infos.values()}) == 1, key
    assert infos["v1"]["tp_size"] == 2 and infos["v1"]["disable_cuda_graph"] is False
    spec = importlib.util.spec_from_file_location("archived_fixed64k_queue", control / "control/fixed64k_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = control
    comparison = module.compare(summaries)
    assert saved_json(comparison) == (control / "comparison.json").read_bytes(), "Full comparison bytes must match"
    # Independently reconstruct key comparisons instead of trusting the producer's compare function.
    base = statistics.mean([summaries["v1"]["mean_output_tps"], summaries["v1-return"]["mean_output_tps"]])
    gain = (summaries["candidate"]["mean_output_tps"] / base - 1) * 100
    assert comparison["combined_v1_tps"] == base and comparison["gain_percent"] == gain
    independent_tails = {}
    for metric in ("ttft_s", "tpot_ms", "e2e_s"):
        for quantile in ("p95", "p99"):
            numerator = statistics.mean(s[metric][quantile] for s in measured["candidate"])
            denominator = statistics.mean([s[metric][quantile] for k in ("v1", "v1-return") for s in measured[k]])
            independent_tails[metric + "_" + quantile] = (numerator / denominator - 1) * 100
    assert comparison["mean_tail_changes_percent"] == independent_tails
    report = dict(status="THREE_ROUND_64K1K_RAW_PERFORMANCE_REVIEW_PASS", run_id=args.run,
                  raw_files=len(manifest), raw_manifest_sha256=sha(manifest_path),
                  queue_source_sha256=sha(control / "control/fixed64k_queue.py"),
                  comparison_sha256=sha(control / "comparison.json"),
                  workload=dict(input_len=65536, output_len=1024, concurrency=8, seed=20260912,
                                physical_pair="4,5", tp=2, warmup_rounds=1, warmup_requests_per_variant=8,
                                measured_rounds_per_variant=3, measured_requests_per_variant=48),
                  measured_requests=144, warmup_requests=24, errors=0,
                  full_raw_sha_pass=True, round_summary_bytes_pass=True,
                  aggregate_summary_bytes_pass=True, independently_validated_summary_bytes_pass=True,
                  full_comparison_bytes_pass=True, same_api_fields=same_api,
                  round_summary_sha256=round_sha, comparison=comparison,
                  timing_scope="Throughputs independently reconstructed from complete JSONL token counts and stored monotonic round wall time; raw clocks cannot be recovered from request rows",
                  input_scope="Original benchmark source/config and same seed/round/request-index generator; no semantic accuracy evaluation",
                  reviewer_repair_note="Original benchmark atomic_json has no trailing newline; initial reviewer A added one, causing byte assertion failure. Raw unchanged; B preserves original serialization.",
                  legacy_metadata_note="Aggregate method text is fixed by archived benchmark and says two warmup waves; config and complete files prove one warmup for 64K",
                  actual_model_graph_gate="REVIEW_REQUIRED", quality_evaluation="NOT_RUN_USER_REQUEST",
                  formal_acceptance=False, frozen=False)
    (output / "comparison-recomputed.json").write_bytes(saved_json(comparison))
    (output / "review.json").write_bytes(saved_json(report))
    print(json.dumps({"status": report["status"], "gain_percent": gain,
                      "mean_tail_5pct_pass": comparison["mean_tail_5pct_pass"],
                      "measured_20pct_pass": comparison["measured_20pct_pass"]}))


if __name__ == "__main__":
    main()
