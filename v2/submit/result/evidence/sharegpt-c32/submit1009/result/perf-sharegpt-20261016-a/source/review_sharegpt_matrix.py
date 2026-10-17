#!/usr/bin/env python3
"""CPU-only review of the 20261016 ShareGPT matrix; never mutates raw data."""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

SCENES = [32, 16, 64, 4, 1]
KINDS = ["v1", "candidate", "v1-return"]
TAILS = ["p99_ttft_ms", "p99_tpot_ms", "p95_itl_ms", "p99_itl_ms", "p99_e2e_latency_ms"]
SERVICE_KEYS = ["model_path", "served_model_name", "tp_size", "pp_size", "context_length",
                "max_running_requests", "max_prefill_tokens", "chunked_prefill_size",
                "mem_fraction_static", "page_size", "disable_radix_cache", "cuda_graph_max_bs",
                "disable_cuda_graph", "speculative_algorithm", "speculative_num_steps",
                "speculative_eagle_topk", "speculative_num_draft_tokens", "mamba_scheduler_strategy",
                "attention_backend", "mm_attention_backend", "kv_cache_dtype", "dtype", "quantization"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def quantile(values, p):
    values = sorted(values)
    assert values
    n = (len(values) - 1) * p
    i = int(n)
    f = n - i
    return values[i] * (1 - f) + values[min(i + 1, len(values) - 1)] * f


def finite(values):
    return all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in values)


def canonical(info):
    info = info.get("server_args", info)
    result = {k: info.get(k) for k in SERVICE_KEYS}
    if result["speculative_algorithm"] == "NEXTN":
        result["speculative_algorithm"] = "EAGLE"
    return result


def benchmark_args(log, c):
    # Installed bench_serving omits seed/num_prompts from result JSON.
    lines = [x for x in log.splitlines() if "benchmark_args" in x]
    assert lines, "Missing benchmark_args parameter evidence"
    text = "\n".join(lines)
    patterns = [r"seed[=:]\s*20260912\b", r"num_prompts[=:]\s*1000\b",
                rf"max_concurrency[=:]\s*{c}\b", r"warmup_requests[=:]\s*64\b",
                r"sharegpt_context_len[=:]\s*67584\b", r"profile[=:]\s*False\b"]
    for pattern in patterns:
        assert re.search(pattern, text), (pattern, text[:1000])
    assert "ShareGPT_V3_unfiltered_cleaned_split.json" in text


def one_result(path, concurrency):
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    assert len(lines) == 1, path
    d = json.loads(lines[0])
    assert d["completed"] == 1000 and d["dataset_name"] == "sharegpt", path
    assert d["max_concurrency"] == concurrency, path
    assert len(d["errors"]) == 1000 and not any(d["errors"]), path
    for key in ["input_lens", "output_lens", "generated_texts", "ttfts", "itls"]:
        assert len(d[key]) == 1000, (path, key)
    assert all(isinstance(n, int) and n > 0 for n in d["input_lens"] + d["output_lens"])
    assert all(isinstance(t, str) and "\ufffd" not in t for t in d["generated_texts"])
    assert finite(d["ttfts"]) and all(finite(v) for v in d["itls"]), path
    assert d["duration"] > 0 and math.isfinite(d["duration"])
    assert d["total_input_tokens"] == sum(d["input_lens"])
    assert d["total_output_tokens"] == sum(d["output_lens"])
    assert math.isclose(d["output_throughput"], sum(d["output_lens"]) / d["duration"], rel_tol=1e-10)
    for key in TAILS:
        assert math.isfinite(d[key]) and d[key] >= 0, (path, key)
    itls = [v * 1000 for row in d["itls"] for v in row]
    # MTP metrics retokenize each text chunk, while the serialized `itls` array
    # contains original streaming-chunk intervals and omits text_chunks. Those
    # intervals cannot reconstruct adjusted per-token ITL percentiles exactly.
    for key, xs, p in [("p99_ttft_ms", [x * 1000 for x in d["ttfts"]], .99)]:
        assert math.isclose(d[key], quantile(xs, p), rel_tol=1e-8, abs_tol=1e-5), (path, key)
    # Serialized output has no per-request E2E latency field. ITL sums omit the
    # final client bookkeeping interval, so report approximate diagnostics only.
    e2e = [(t + sum(ils)) * 1000 for t, ils in zip(d["ttfts"], d["itls"])]
    tpot = [sum(ils) * 1000 / (n - 1) for ils, n in zip(d["itls"], d["output_lens"]) if n > 1]
    residual = {"e2e_p99_reported_minus_itl_reconstruction_ms": d["p99_e2e_latency_ms"] - quantile(e2e, .99),
                "tpot_p99_reported_minus_itl_reconstruction_ms": d["p99_tpot_ms"] - quantile(tpot, .99),
                "raw_chunk_itl_p95_ms_not_adjusted_per_token_metric": quantile(itls,.95),
                "raw_chunk_itl_p99_ms_not_adjusted_per_token_metric": quantile(itls,.99)}
    benchmark_args(path.with_suffix(".log").read_text(errors="replace"), concurrency)
    return d, residual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", default="perf-sharegpt-20261016-a")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    assert not a.out.exists(), "Refusing to overwrite review output"
    run_dir = a.root / "result" / a.run
    manifest = read(run_dir / "artifact-sha256.json")
    for relative, expected in manifest.items():
        path = a.root / relative
        assert os.path.commonpath([str(path.resolve()),str(a.root.resolve())])==str(a.root.resolve()),relative
        assert sha(path) == expected, relative
    summaries, stages, all_input_lens = {}, {}, None
    for c in SCENES:
        per_kind = {}
        common = None
        for kind in KINDS:
            meta_path = run_dir / "stages" / f"c{c}-{kind}.json"
            if not meta_path.exists():
                continue
            stage = read(meta_path)
            assert stage["status"] == "COMPLETE" and stage["physical_pair"] == [6, 7]
            assert stage["client_concurrency"] == c and stage["kind"] == kind
            evidence = a.root / stage["evidence_dir"]
            stage_paths = [str(meta_path.relative_to(a.root))]
            stage_paths += [str(x.relative_to(a.root)) for x in evidence.rglob("*") if x.is_file()]
            directory = a.root / stage["result_dir"]
            stage_paths += [str(x.relative_to(a.root)) for x in directory.rglob("*") if x.is_file()]
            assert all(x in manifest for x in stage_paths), "Stage artifacts omitted from manifest"
            info = read(evidence / "server-info.json")
            parameters = canonical(info)
            assert parameters["tp_size"] == 2 and parameters["context_length"] == 67584
            assert parameters["max_running_requests"] == 32
            assert parameters["cuda_graph_max_bs"] == 32 and not parameters["disable_cuda_graph"]
            assert parameters["speculative_algorithm"] == "EAGLE"
            if common is None:
                common = parameters
            assert parameters == common, f"Service parameters changed at c{c}/{kind}"
            server_args = info.get("server_args", info)
            log = (evidence / "server.log").read_text(errors="replace")
            if kind == "candidate":
                assert server_args.get("custom_all_reduce_backend") == "native"
                assert server_args.get("disable_custom_all_reduce") is False
                assert stage["native_guard_verified"] is True
                assert len(set(re.findall(r"\[Native BF16 AR\] Eligible rank=(\d+)", log))) == 2
            else:
                assert server_args.get("disable_custom_all_reduce") is True
                assert server_args.get("custom_all_reduce_backend") in ["off", None]
            measured, residuals = [], []
            for label in ["full-warmup-01", "round-01", "round-02", "round-03"]:
                d, residual = one_result(directory / f"{label}.jsonl", c)
                if all_input_lens is None:
                    all_input_lens = d["input_lens"]
                assert d["input_lens"] == all_input_lens, "Request set differs between rounds/scenes/variants"
                residuals.append(dict(round=label, **residual))
                if label.startswith("round"):
                    measured.append(d)
            assert len(list(directory.glob("round-*.jsonl"))) == 3
            assert len(list(directory.glob("full-warmup-*.jsonl"))) == 1
            original = read(directory / "validated_summary.json")
            vs = [x["output_throughput"] for x in measured]
            assert original["status"] == "PASS" and original["requests"] == 3000
            assert original["output_tps_rounds"] == vs
            assert original["full_workload_warmup_rounds"] == 1
            assert math.isclose(original["mean_output_tps"], statistics.mean(vs), rel_tol=1e-12)
            rebuilt=dict(status="PASS",rounds=3,requests=3000,errors=0,
                full_workload_warmup_rounds=1,short_warmup_requests_per_run=64,
                client_concurrency=c,output_tps_rounds=vs,mean_output_tps=statistics.mean(vs),
                spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100,
                mean_duration_s=statistics.mean(x["duration"] for x in measured))
            for key in ["median_ttft_ms","p99_ttft_ms","median_tpot_ms","p99_tpot_ms",
                        "median_itl_ms","p95_itl_ms","p99_itl_ms","p99_e2e_latency_ms"]:
                values=[x[key] for x in measured]
                assert finite(values),key
                rebuilt[key+"_mean_of_rounds"]=statistics.mean(values)
                rebuilt[key+"_range"]=[min(values),max(values)]
            assert (json.dumps(rebuilt,indent=2)+"\n").encode()==(directory/"validated_summary.json").read_bytes(),"Independent summary bytes differ"
            per_kind[kind] = dict(mean_output_tps=statistics.mean(vs), output_tps_rounds=vs,
                spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100, errors=0, requests=3000,
                tail_mean={k: statistics.mean(x[k] for x in measured) for k in TAILS},
                tail_rounds={k: [x[k] for x in measured] for k in TAILS},
                reconstruction_diagnostics=residuals, parameters=parameters)
            stages[f"c{c}-{kind}"] = stage
        if len(per_kind) != 3:
            if not a.allow_partial:
                raise AssertionError(f"Incomplete c{c} matrix: {list(per_kind)}")
            continue
        v, b, r = (per_kind[k] for k in KINDS)
        base = statistics.mean(v["output_tps_rounds"] + r["output_tps_rounds"])
        paired_base = [(x+y)/2 for x,y in zip(v["output_tps_rounds"],r["output_tps_rounds"])]
        gain = (b["mean_output_tps"]/base-1)*100
        drift = (r["mean_output_tps"]/v["mean_output_tps"]-1)*100
        tails = {k: (b["tail_mean"][k] / statistics.mean(v["tail_rounds"][k]+r["tail_rounds"][k])-1)*100 for k in TAILS}
        tail_pairs = {k: [(z/((x+y)/2)-1)*100 for x,y,z in zip(v["tail_rounds"][k],r["tail_rounds"][k],b["tail_rounds"][k])] for k in TAILS}
        noise = max(abs(drift), *(s["spread_percent"] for s in per_kind.values()))
        summaries[str(c)] = dict(variants=per_kind, baseline_combined_mean_output_tps=base,
            candidate_gain_percent=gain, paired_gain_percent=[(x/y-1)*100 for x,y in zip(b["output_tps_rounds"],paired_base)],
            v1_return_drift_percent=drift, observed_variation_reference_percent=noise,
            gain_exceeds_observed_variation=gain>noise, meets_20_percent=gain>=20,
            tail_change_percent=tails, tail_paired_change_percent=tail_pairs,
            tail_regression_le_5_percent=all(x<=5 for x in tails.values()) and all(x<=5 for xs in tail_pairs.values() for x in xs),
            comparison_scope="Same physical HCU6,7, TP2, concurrency and request set; overlap retained in stage metadata")
    assert summaries, "No complete scenario to review"
    if not a.allow_partial:
        assert len(summaries) == len(SCENES)
    result = dict(status="SHAREGPT_PERFORMANCE_RAW_REVIEW_PASS", run=a.run, scenarios=summaries,
        complete=len(summaries)==len(SCENES), manifest_sha256=sha(run_dir/"artifact-sha256.json"),
        artifact_count=len(manifest), stages=stages, accuracy="NOT_RUN", default_patch_changed=False,
        frozen=False, actual_model_graph_gate="REVIEW_REQUIRED",
        caveats=["Observed variation is a reference, not a statistical noise upper bound.",
                 "TTFT quantiles and output throughput are reconstructed exactly. ITL/TPOT/E2E retain the installed benchmark metrics and are independently aggregated/compared; text_chunks and per-request final latency are not serialized, so per-token ITL and exact E2E/TPOT cannot be reconstructed. Raw-chunk ITL and interval-sum diagnostics are separate and approximate.",
                 "This performance validation neither runs accuracy nor promotes/freezes the patch."])
    a.out.mkdir(parents=True)
    write(a.out/"comparison.json", result)
    write(a.out/"review-sha256.json", {"comparison.json":sha(a.out/"comparison.json")})
    print(json.dumps(dict(status=result["status"],complete=result["complete"],
        gains={c:s["candidate_gain_percent"] for c,s in summaries.items()})))


if __name__ == "__main__":
    main()
