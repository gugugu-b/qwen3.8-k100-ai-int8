#!/usr/bin/env python3
"""CPU-only review of latest submit C64, server64/Graph64, physical pair0,1.

Default: strict completed-scene review for queue producers. --allow-partial
allows a genuinely incomplete scene report. --final-controls additionally
requires all terminal control files and exact artifact/control coverage.
No GPU, network, container actions, source writes, or existing-output writes.
"""
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



RUN = "submit-current-c64-server64-pair01-20261016-a"
RUNTIME_REL = "experiments/performance-validation-c64-server64-20261016"
SITE_SHA = "da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610"
GUARD_SHA = "f9d5ec98c70724dcb80857228681a6c03c619606643cdd6918c9084af51cd7f4"
DATASET_SHA = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
OLD_SERVER32_TPS = 417.09167316715815
MUTABLE = {"artifact-sha256.json", "performance-summary.json", "finished", "final-state.json", "failed-state.json", "control-sha256.json"}


def manifest_file(root, relative):
    assert isinstance(relative, str) and relative
    rel = Path(relative)
    assert not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == relative, relative
    path = root / relative
    assert os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve()), relative
    assert path.is_file(), relative
    return path


def verify_manifest(root, manifest):
    assert isinstance(manifest, dict) and manifest
    for relative, expected in manifest.items():
        assert isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), relative
        assert sha(manifest_file(root, relative)) == expected, relative


def exact_artifact_paths(root, run_dir, stage):
    paths = [x for x in run_dir.rglob("*") if x.is_file() and x.name not in MUTABLE
             and not any(part.startswith("review-") for part in x.relative_to(run_dir).parts)]
    if stage is not None:
        paths.extend(x for x in (root / stage["result_dir"]).rglob("*") if x.is_file())
    raw = root / RUNTIME_REL / "result" / (RUN + "-c64-candidate") / "sharegpt-c64"
    if raw.exists():
        paths.extend(x for x in raw.rglob("*") if x.is_file())
    return {str(x.relative_to(root)) for x in paths}


def terminal_controls(root, run_dir, artifact, required):
    path = run_dir / "control-sha256.json"
    if not path.exists():
        assert not required, "Final control manifest is required"
        return {"required": required, "present": False, "verified": False}
    control = read(path)
    verify_manifest(root, control)
    expected = {str(x.relative_to(root)) for x in run_dir.rglob("*")
                if x.is_file() and x.name != "control-sha256.json"}
    assert set(control) == expected, "Exact final control coverage differs"
    assert (run_dir / "finished").is_file() and (run_dir / "finished").read_text().strip()
    assert not (run_dir / "failed-state.json").exists(), "Original queue failed"
    state = read(run_dir / "final-state.json")
    assert state["run"] == RUN and state["physical_pair"] == [0, 1]
    assert state["phase"] == "submit-c64-server64-pair01-complete"
    assert state["accuracy"] == "NOT_RUN" and state["frozen"] is False
    assert state["default_patch_changed"] is False
    release = read(run_dir / "card-release.json")
    assert release["queue_success"] is True and release["running"] is False
    assert release["physical_pair"] == [0, 1]
    assert release["container"] == "qwen38-submit1009-perf01-server64-20261016"
    assert (run_dir / "review-final/performance-summary.json").read_bytes() == (run_dir / "performance-summary.json").read_bytes()
    external = root / "result" / (RUN + "-state.json")
    if external.exists():
        assert external.read_bytes() == (run_dir / "final-state.json").read_bytes()
    return {"required": required, "present": True, "verified": True,
            "control_count": len(control), "control_sha256": sha(path),
            "finished": True, "final_phase": state["phase"], "owned_container_stopped_by_queue": True}


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
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--run",default=RUN)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--final-controls",action="store_true",help="Require complete immutable terminal controls; use outside original RUN")
    a=p.parse_args()
    pair=[0,1]
    scenes=[64]
    run=a.run
    assert run==RUN,"Only the explicitly authorized pair01 server64 RUN is accepted"
    assert not (a.final_controls and a.allow_partial),"Final controls require complete strict review"
    assert not a.out.exists(),"Refusing overwrite of review output"
    run_dir=a.root/"result"/run
    manifest=read(run_dir/"artifact-sha256.json")
    verify_manifest(a.root,manifest)
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
    assert reference["sharegpt"]["64"]["output_tps"]==417.42
    assert preflight["prompts_per_round"]==1000 and preflight["short_warmup_per_run"]==64
    assert preflight["context"]==67584 and preflight["server_max_running_requests"]==64
    assert preflight["cuda_graph_max_bs"]==64
    assert preflight["container"]=="qwen38-submit1009-perf01-server64-20261016"
    assert preflight["port"]==31034 and preflight["nccl_port"]==31035
    assert preflight["image"]=="sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
    assert preflight["dataset_path"]=="/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    assert preflight["dataset_sha256"]==DATASET_SHA
    runtime_path=run_dir/"source/runtime-manifest.json"
    runtime=read(runtime_path)
    assert sha(runtime_path)==preflight["runtime_manifest_sha256"]
    assert runtime["dataset_sha256"]==DATASET_SHA
    assert preflight["runtime_files_verified"]==len(runtime["files"])
    candidate=run_dir/"source/candidate01"
    assert sha(candidate/"sitecustomize.py")==SITE_SHA
    assert sha(candidate/"native_bf16_ar_guard.py")==GUARD_SHA
    candidate_manifest=read(candidate/"native-ar-manifest.json")
    for relative,expected in candidate_manifest["files"].items():
        assert sha(manifest_file(candidate,relative))==expected,relative
    candidate_files=preflight["candidate_sha256"]
    assert candidate_files
    for relative,expected in candidate_files.items():
        prefix=RUNTIME_REL+"/candidate01/"
        assert relative.startswith(prefix),relative
        assert sha(manifest_file(candidate,relative[len(prefix):]))==expected,relative
    for name in ["run_server64_pair01.py","review_server64_pair01.py","scripts/start_service.sh", "scripts/serve.sh", "scripts/benchmark_sharegpt.sh", "scripts/validate_sharegpt.py"]:
        path=run_dir/"source"/name
        assert path.is_file() and str(path.relative_to(a.root)) in manifest,name
    for field,name in [("reviewer_source_sha256","review_server64_pair01.py"),("queue_source_sha256","run_server64_pair01.py")]:
        if field in preflight:
            assert sha(run_dir/"source"/name)==preflight[field],field
    gatepath=run_dir/"source/candidate01-gate.json"
    gate=read(gatepath)
    assert gate["status"]=="NUMERICAL_GRAPH_PASS" and gate["physical_pair"]=="0,1"
    assert sha(gatepath)==preflight["numerical_gate_sha256"]
    assert gate["candidate_site_sha256"]==SITE_SHA and gate["candidate_guard_sha256"]==GUARD_SHA
    reports={};all_inputs=None;common_parameters=None
    for c in scenes:
        metapath=run_dir/"stages"/f"c{c}-candidate.json"
        if not metapath.exists():
            assert a.allow_partial,f"Missing complete c{c} candidate stage"
            continue
        stage=read(metapath)
        assert stage["status"]=="COMPLETE" and stage["kind"]=="candidate"
        assert stage["physical_pair"]==pair and stage["client_concurrency"]==c
        assert stage["server_max_running_requests"]==64 and stage["cuda_graph_max_bs"]==64
        assert stage["result_dir"]==RUNTIME_REL+"/result/"+RUN+"-c64-candidate/sharegpt-c64"
        assert stage["evidence_dir"]=="result/"+RUN+"/evidence/c64-candidate"
        evidence=a.root/stage["evidence_dir"];directory=a.root/stage["result_dir"]
        required=[metapath]+[x for x in evidence.rglob("*") if x.is_file()]+[x for x in directory.rglob("*") if x.is_file()]
        assert all(str(x.relative_to(a.root)) in manifest for x in required),"Missing manifest coverage"
        container=read(evidence/"container-inspect.json")
        assert container["Name"].lstrip("/")=="qwen38-submit1009-perf01-server64-20261016"
        assert container["Image"]==preflight["image"]
        assert container["Config"].get("Labels",{}).get("qwen38.submit1009.performance")=="20261016"
        assert "HIP_VISIBLE_DEVICES=0,1" in container["Config"]["Env"]
        info=read(evidence/"server-info.json");args=info.get("server_args",info)
        parameters=canonical(info)
        expected={"tp_size":2,"pp_size":1,"context_length":67584,"mem_fraction_static":.85,
            "max_running_requests":64,"max_prefill_tokens":65536,"chunked_prefill_size":8192,
            "mamba_scheduler_strategy":"no_buffer","cuda_graph_max_bs":64,"disable_cuda_graph":False,
            "disable_radix_cache":True,"attention_backend":"fa3","mm_attention_backend":"fa3",
            "page_size":64,"speculative_num_steps":1,"speculative_eagle_topk":1,"speculative_num_draft_tokens":2}
        for key,value in expected.items():assert parameters[key]==value,(key,parameters[key],value)
        assert parameters["speculative_algorithm"]=="EAGLE"
        assert parameters["model_path"]=="/data/model/Qwen3.8-27B-Channel-INT8-w8a8"
        assert parameters["served_model_name"]=="Qwen3.8-27B-Channel-INT8-w8a8"
        assert parameters["dtype"]=="auto" and parameters["kv_cache_dtype"]=="auto"
        assert parameters["quantization"] is None
        assert args["custom_all_reduce_backend"]=="native" and args["disable_custom_all_reduce"] is False
        if common_parameters is None:common_parameters=parameters
        assert parameters==common_parameters,"Service parameters changed across scenes"
        serverlog=(evidence/"server.log").read_text(errors="replace")
        assert stage["native_guard_verified"] is True
        assert serverlog.count("[Native BF16 AR] Installed")>=2
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
        reports[str(c)]["prior_server32_descriptive_comparison"]=dict(
            prior_physical_pair=[6,7],prior_client_concurrency=64,
            prior_server_max_running_requests=32,prior_cuda_graph_max_bs=32,
            prior_mean_output_tps=OLD_SERVER32_TPS,latest_mean_output_tps=statistics.mean(vs),
            output_tps_change_percent=(statistics.mean(vs)/OLD_SERVER32_TPS-1)*100,
            comparison_scope="Different physical pair and time, plus server/Graph64 configuration change; descriptive only, not causal A/B",
            formal_acceptance=False)
        reports[str(c)]["server_max_running_requests"]=64
        reports[str(c)]["cuda_graph_max_bs"]=64
    if not reports:
        assert a.allow_partial,"No completed submit scenario"
    stagefile=run_dir/"stages/c64-candidate.json"
    stage=read(stagefile) if stagefile.exists() else None
    assert set(manifest)==exact_artifact_paths(a.root,run_dir,stage),"Exact artifact coverage differs"
    controls=terminal_controls(a.root,run_dir,manifest,a.final_controls)
    if a.final_controls:
        assert len(reports)==1,"Final report incomplete"
        assert not os.path.commonpath([str(a.out.resolve()),str(run_dir.resolve())])==str(run_dir.resolve()),"Final independent review must live outside original RUN"

    if not a.allow_partial:assert len(reports)==len(scenes)
    result=dict(status="LATEST_SUBMIT_SERVER64_PAIR01_ABSOLUTE_PERFORMANCE_RAW_REVIEW_PASS",run=run,physical_pair=pair,
        complete=len(reports)==len(scenes),scenarios=reports,manifest_sha256=sha(run_dir/"artifact-sha256.json"),
        artifact_count=len(manifest),accuracy="NOT_RUN",default_patch_changed=False,frozen=False,
        acceptance_comparison="NOT_REQUESTED_CANDIDATE_ONLY",actual_model_graph_gate="REVIEW_REQUIRED",
        server_max_running_requests=64,cuda_graph_max_bs=64,client_concurrency=64,
        caveats=["Only the latest submit candidate is measured. No new v1 measurements are run. A separate comparison to the user-selected historical report is descriptive, not contemporaneous same-pair A/B or acceptance evidence.",
                 "The prior server32 C64 comparison uses physical6,7 versus current0,1 at another time and changes server/Graph limits; it is descriptive only, not isolated causal evidence.",
                 "Physical0,1 were explicitly authorized for this independent new RUN; original4–7 completed results are retained unchanged.",
                 "Three-round spread is observed variability, not a statistical noise upper bound.",
                 "TTFT P99 and output throughput are reconstructed exactly. ITL/TPOT/E2E retain installed benchmark metrics and are independently aggregated; MTP text_chunks and per-request final latency are not serialized, so those exact per-request tails cannot be reconstructed.",
                 "Round percentile means are not pooled-request percentiles; raw chunk ITL and interval-sum reconstructions are approximate diagnostics.",
                 "Workload SHA/seed/input-length sequence are checked; no semantic accuracy evaluation, quality test, patch promotion or freeze is performed."])
    a.out.mkdir(parents=True)
    write(a.out/"performance-summary.json",result)
    write(a.out/"manifest-review.json",dict(status="MANIFEST_SHA_AND_COVERAGE_REVIEW_PASS",
        run=RUN,physical_pair=pair,artifact_sha256=sha(run_dir/"artifact-sha256.json"),
        artifact_count=len(manifest),exact_artifact_coverage=True,final_controls=controls,
        note="Final control proof is separate from stable performance-summary bytes; no GPU or live-process checks in this CPU reviewer."))
    write(a.out/"review-sha256.json",{name:sha(a.out/name) for name in ["performance-summary.json","manifest-review.json"]})
    print(json.dumps(dict(status=result["status"],complete=result["complete"],
        throughput={c:v["mean_output_tps"] for c,v in reports.items()})))


if __name__=="__main__":main()

