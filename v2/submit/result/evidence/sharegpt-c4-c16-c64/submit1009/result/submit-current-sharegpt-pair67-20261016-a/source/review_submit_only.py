#!/usr/bin/env python3
"""CPU-only validation of latest submit absolute performance; no baseline comparisons."""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

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
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--pair45",action="store_true")
    modes.add_argument("--pair67",action="store_true")
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--run")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args()
    pair=[4,5] if a.pair45 else [6,7]
    scenes=[1] if a.pair45 else [16,64,4]
    expected_run="submit-current-c1-pair45-20261016-a" if a.pair45 else "submit-current-sharegpt-pair67-20261016-a"
    run=a.run or expected_run
    assert not a.out.exists(),"Refusing overwrite of review output"
    run_dir=a.root/"result"/run
    manifest=read(run_dir/"artifact-sha256.json")
    for relative,expected in manifest.items():
        path=a.root/relative
        assert os.path.commonpath([str(path.resolve()),str(a.root.resolve())])==str(a.root.resolve())
        assert sha(path)==expected,relative
    preflight=read(run_dir/"preflight.json")
    assert preflight["run"]==run and preflight["physical_pair"]==pair
    assert preflight["scenes"]==scenes and preflight["kinds"]==["candidate"]
    assert preflight["seed"]==20260912 and preflight["full_warmup_rounds"]==1
    assert preflight["measure_rounds"]==3 and preflight["accuracy"]=="NOT_RUN"
    reference_path=run_dir/"source/v1-report-reference.json"
    source_path=run_dir/"source/v1-report-source.md"
    assert sha(reference_path)==preflight["historical_report_reference_sha256"]=="54c5ff586396dd90760eeb3fc3686f741927a5ef065294148ce293919ab4ef2d"
    assert sha(source_path)==preflight["historical_report_source_sha256"]=="b36db6ec60fcac45d710fada2bdf74b1b4655ef631567e6c34fc4dc4a20e417d"
    reference=read(reference_path)
    if a.pair45:
        gatepath=run_dir/"source/candidate45-gate.json"
        gate=read(gatepath)
        assert gate["status"]=="NUMERICAL_GRAPH_PASS" and gate["physical_pair"]=="4,5"
        assert sha(gatepath)==preflight["numerical_gate_sha256"]
    else:
        proof=run_dir/"source/prior-c32-candidate-runtime-evidence.json"
        d=read(proof)
        assert d["status"]=="COMPLETE" and d["physical_pair"]==[6,7]
        assert d["kind"]=="candidate" and d["native_guard_verified"] is True
        assert sha(proof)==preflight["prior_runtime_compatibility_sha256"]
    reports={};all_inputs=None;common_parameters=None
    for c in scenes:
        metapath=run_dir/"stages"/f"c{c}-candidate.json"
        if not metapath.exists():
            assert a.allow_partial,f"Missing complete c{c} candidate stage"
            continue
        stage=read(metapath)
        assert stage["status"]=="COMPLETE" and stage["kind"]=="candidate"
        assert stage["physical_pair"]==pair and stage["client_concurrency"]==c
        evidence=a.root/stage["evidence_dir"];directory=a.root/stage["result_dir"]
        required=[metapath]+[x for x in evidence.rglob("*") if x.is_file()]+[x for x in directory.rglob("*") if x.is_file()]
        assert all(str(x.relative_to(a.root)) in manifest for x in required),"Missing manifest coverage"
        info=read(evidence/"server-info.json");args=info.get("server_args",info)
        parameters=canonical(info)
        expected={"tp_size":2,"pp_size":1,"context_length":67584,"mem_fraction_static":.85,
            "max_running_requests":32,"max_prefill_tokens":65536,"chunked_prefill_size":8192,
            "mamba_scheduler_strategy":"no_buffer","cuda_graph_max_bs":32,"disable_cuda_graph":False,
            "disable_radix_cache":True,"attention_backend":"fa3","mm_attention_backend":"fa3",
            "page_size":64,"speculative_num_steps":1,"speculative_eagle_topk":1,"speculative_num_draft_tokens":2}
        for key,value in expected.items():assert parameters[key]==value,(key,parameters[key],value)
        assert parameters["speculative_algorithm"]=="EAGLE"
        assert args["custom_all_reduce_backend"]=="native" and args["disable_custom_all_reduce"] is False
        if common_parameters is None:common_parameters=parameters
        assert parameters==common_parameters,"Service parameters changed across scenes"
        serverlog=(evidence/"server.log").read_text(errors="replace")
        assert stage["native_guard_verified"] is True
        assert {0,1}<={int(v) for v in re.findall(r"\[Native BF16 AR\] Eligible rank=(\d+)",serverlog)}
        measured=[];diagnostics=[]
        for label in ["full-warmup-01","round-01","round-02","round-03"]:
            d,residual=one_result(directory/f"{label}.jsonl",c)
            if all_inputs is None:all_inputs=d["input_lens"]
            assert d["input_lens"]==all_inputs,"Input length sequences differ"
            diagnostics.append(dict(round=label,**residual))
            if label.startswith("round"):measured.append(d)
        assert len(list(directory.glob("round-*.jsonl")))==3
        assert len(list(directory.glob("full-warmup-*.jsonl")))==1
        vs=[x["output_throughput"] for x in measured]
        rebuilt=dict(status="PASS",rounds=3,requests=3000,errors=0,
            full_workload_warmup_rounds=1,short_warmup_requests_per_run=64,client_concurrency=c,
            output_tps_rounds=vs,mean_output_tps=statistics.mean(vs),
            spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100,
            mean_duration_s=statistics.mean(x["duration"] for x in measured))
        for key in ["median_ttft_ms","p99_ttft_ms","median_tpot_ms","p99_tpot_ms",
                    "median_itl_ms","p95_itl_ms","p99_itl_ms","p99_e2e_latency_ms"]:
            values=[x[key] for x in measured];assert finite(values)
            rebuilt[key+"_mean_of_rounds"]=statistics.mean(values)
            rebuilt[key+"_range"]=[min(values),max(values)]
        assert (json.dumps(rebuilt,indent=2)+"\n").encode()==(directory/"validated_summary.json").read_bytes(),"Summary independent rebuild differs"
        reports[str(c)]=dict(status="LATEST_SUBMIT_SCENARIO_PERFORMANCE_VALIDATED",physical_pair=pair,
            client_concurrency=c,rounds=3,measured_requests=3000,full_workload_warmup_requests=1000,
            short_warmup_requests_per_invocation=64,benchmark_invocations=4,errors=0,
            output_tps_rounds=vs,mean_output_tps=statistics.mean(vs),
            spread_percent=(max(vs)-min(vs))/statistics.mean(vs)*100,
            total_input_tokens_rounds=[x["total_input_tokens"] for x in measured],
            total_output_tokens_rounds=[x["total_output_tokens"] for x in measured],
            tail_mean={key:statistics.mean(x[key] for x in measured) for key in TAILS},
            tail_rounds={key:[x[key] for x in measured] for key in TAILS},
            reconstruction_diagnostics=diagnostics,parameters=parameters,stage=stage)
        old=reference["sharegpt"][str(c)]
        reports[str(c)]["historical_report_reference_comparison"]=dict(
            source_user_reference=reference["source_user_reference"],source_sha256=reference["source_sha256"],
            reference_output_tps=old["output_tps"],latest_mean_output_tps=statistics.mean(vs),
            output_tps_change_percent=(statistics.mean(vs)/old["output_tps"]-1)*100,
            tail_latest_mean={k:statistics.mean(x[k] for x in measured) for k in old if k!="output_tps"},
            tail_reference={k:v for k,v in old.items() if k!="output_tps"},
            tail_change_percent={k:(statistics.mean(x[k] for x in measured)/v-1)*100 for k,v in old.items() if k!="output_tps"},
            comparison_scope="User-selected historical v1 report reference only; not contemporaneous same-pair A/B or acceptance evidence",
            formal_acceptance=False)
    assert reports,"No completed submit scenario"
    if not a.allow_partial:assert len(reports)==len(scenes)
    result=dict(status="LATEST_SUBMIT_ABSOLUTE_PERFORMANCE_RAW_REVIEW_PASS",run=run,physical_pair=pair,
        complete=len(reports)==len(scenes),scenarios=reports,manifest_sha256=sha(run_dir/"artifact-sha256.json"),
        artifact_count=len(manifest),accuracy="NOT_RUN",default_patch_changed=False,frozen=False,
        acceptance_comparison="NOT_REQUESTED_CANDIDATE_ONLY",actual_model_graph_gate="REVIEW_REQUIRED",
        caveats=["Only the latest submit candidate is measured. No new v1 measurements are run. A separate comparison to the user-selected historical report is descriptive, not contemporaneous same-pair A/B or acceptance evidence.",
                 "Three-round spread is observed variability, not a statistical noise upper bound.",
                 "TTFT P99 and output throughput are reconstructed exactly. ITL/TPOT/E2E retain installed benchmark metrics and are independently aggregated; MTP text_chunks and per-request final latency are not serialized, so those exact per-request tails cannot be reconstructed.",
                 "Round percentile means are not pooled-request percentiles; raw chunk ITL and interval-sum reconstructions are approximate diagnostics.",
                 "Workload SHA/seed/input-length sequence are checked; no semantic accuracy evaluation, quality test, patch promotion or freeze is performed."])
    a.out.mkdir(parents=True)
    write(a.out/"performance-summary.json",result)
    write(a.out/"review-sha256.json",{"performance-summary.json":sha(a.out/"performance-summary.json")})
    print(json.dumps(dict(status=result["status"],complete=result["complete"],
        throughput={c:v["mean_output_tps"] for c,v in reports.items()})))


if __name__=="__main__":main()

