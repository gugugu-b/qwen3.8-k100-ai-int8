#!/usr/bin/env python3
"""Latest-submit HCU0,1 TP2 server64/client64 ShareGPT performance validation, no accuracy/promotion."""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/wangzy/submit1009")
RUNTIME = ROOT/"experiments/performance-validation-c64-server64-20261016"
RUN = "submit-current-c64-server64-pair01-20261016-a"
PAIR = [0,1]
PAIR_ID = "01"
CONTAINER = "qwen38-submit1009-perf01-server64-20261016"
PORT = 31034
STATE = ROOT/"result"/f"{RUN}-state.json"
OUT = ROOT/"result"/RUN
IMAGE = "sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
SCENES = [64]
KINDS = ["candidate"]
CURRENT = {}
ACTIVE_SERVER = None
ACTIVE_EVIDENCE = None
START = time.monotonic()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def state(phase,**kw):
    global CURRENT
    CURRENT=dict(phase=phase,pid=os.getpid(),utc=now(),run=RUN,physical_pair=PAIR,server_max_running_requests=64,cuda_graph_max_bs=64,
                 elapsed_s=time.monotonic()-START,**kw)
    tmp=STATE.with_suffix(".tmp")
    write(tmp,CURRENT);tmp.replace(STATE)
    print(json.dumps(CURRENT),flush=True)


def run(args,timeout=180):
    return subprocess.run(args,check=True,timeout=timeout,text=True,capture_output=True).stdout


def owned():
    o=json.loads(run(["docker","inspect",CONTAINER]))[0]
    assert o["Image"]==IMAGE
    assert "HIP_VISIBLE_DEVICES="+",".join(map(str,PAIR)) in o["Config"]["Env"]
    assert o["Config"].get("Labels",{}).get("qwen38.submit1009.performance")=="20261016"
    assert o["Config"].get("Labels",{}).get("qwen38.submit1009.server64")=="pair01"
    assert any(x["Source"]==str(ROOT) and x["Destination"]==str(ROOT) for x in o["Mounts"])
    return o


def overlap():
    result={"utc":now()}
    for name in ["perf-sharegpt-20261016-a-state.json", "perf-sharegpt-c1-pair45-20261016-a-state.json", "perf-64k-20261016-a-state.json", "submit-current-c1-pair45-20261016-a-state.json", "submit-current-sharegpt-pair67-20261016-a-state.json"]:
        p=ROOT/"result"/name
        if p.exists() and p!=STATE:
            result[name]=json.loads(p.read_text())
    for name in ["qwen38-submit1009-perf45-20261016","qwen38-submit1009-perf67-20261016"]:
        obj=subprocess.run(["docker","inspect",name],capture_output=True,text=True,timeout=20)
        if obj.returncode==0:
            d=json.loads(obj.stdout)[0]
            result[name]={"running":d["State"]["Running"],"pid":d["State"]["Pid"]}
    return result


def command(args,log,timeout,phase,**kw):
    assert not log.exists(),log
    with log.open("w") as stream:
        proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state(phase,command_pid=proc.pid,log=str(log.relative_to(ROOT)),overlap=overlap(),**kw)
        deadline=time.monotonic()+timeout
        try:
            while proc.poll() is None:
                assert time.monotonic()<deadline,"Command timeout"
                assert time.monotonic()-START<36*3600,"Queue exceeded 36h"
                time.sleep(5)
            assert proc.returncode==0,dict(command=args,exit_code=proc.returncode,log=str(log))
        except BaseException:
            # Only the newly spawned host docker-exec/launcher group.
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
            raise


def get(path):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{PORT}/{path}",timeout=10) as r:
        body=r.read()
        return json.loads(body) if path!="health" else r.status


def api(kind):
    assert kind=="candidate","Latest submit validation never starts baseline"
    deadline=time.monotonic()+900
    while True:
        try:
            assert get("health")==200
            info=get("get_server_info")
            args=info.get("server_args",info)
            expected={"tp_size":2,"pp_size":1,"context_length":67584,
                "mem_fraction_static":.85,"max_running_requests":64,"max_prefill_tokens":65536,
                "chunked_prefill_size":8192,"mamba_scheduler_strategy":"no_buffer",
                "cuda_graph_max_bs":64,"disable_cuda_graph":False,"disable_radix_cache":True,
                "attention_backend":"fa3","mm_attention_backend":"fa3","page_size":64,
                "speculative_num_steps":1,"speculative_eagle_topk":1,"speculative_num_draft_tokens":2}
            for key,value in expected.items():assert args.get(key)==value,(key,args.get(key),value)
            assert args["speculative_algorithm"] in ["NEXTN","EAGLE"]
            assert args["custom_all_reduce_backend"]=="native" and args["disable_custom_all_reduce"] is False
            return info
        except (OSError,ValueError):
            assert time.monotonic()<deadline,"Service readiness timeout"
            time.sleep(10)


def manifest():
    mutable={"artifact-sha256.json","performance-summary.json","finished","final-state.json","failed-state.json","control-sha256.json"}
    paths=[p for p in OUT.rglob("*") if p.is_file() and p.name not in mutable and not any(x.startswith("review-") for x in p.relative_to(OUT).parts)]
    for stagefile in sorted((OUT/"stages").glob("*.json")):
        stage=json.loads(stagefile.read_text())
        paths.extend(p for p in (ROOT/stage["result_dir"]).rglob("*") if p.is_file())
    # Preserve partial client JSONL/logs on failure too, before a COMPLETE
    # stage metadata file can exist. Only this independent RUN's candidate paths.
    for c in SCENES:
        raw=RUNTIME/"result"/f"{RUN}-c{c}-candidate"/f"sharegpt-c{c}"
        if raw.exists():paths.extend(p for p in raw.rglob("*") if p.is_file())
    values={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    write(OUT/"artifact-sha256.json",values)


def review(name,partial):
    args=["python3","-B",str(RUNTIME/"review_server64_pair01.py"),"--root",str(ROOT),"--run",RUN,"--out",str(OUT/name)]
    if partial:args.append("--allow-partial")
    command(args,OUT/f"{name}.log",3600,"cpu-review",review=name)
    summary=json.loads((OUT/name/"performance-summary.json").read_text())
    write(OUT/"performance-summary.json",summary)


def stage(c,kind):
    global ACTIVE_SERVER,ACTIVE_EVIDENCE
    subrun=f"{RUN}-c{c}-{kind}"
    serverrun=subrun+"-server"
    results=RUNTIME/"result"/subrun/f"sharegpt-c{c}"
    evidence=OUT/"evidence"/f"c{c}-{kind}"
    assert not results.exists() and not evidence.exists()
    evidence.mkdir(parents=True)
    ACTIVE_SERVER=serverrun;ACTIVE_EVIDENCE=evidence
    started=now();before=overlap()
    owned()
    state("starting-service",client_concurrency=c,kind=kind,server_run=serverrun,overlap=before)
    run(["docker","restart",CONTAINER],180)
    patch=RUNTIME/("candidate"+PAIR_ID)
    assert kind=="candidate"
    cmd=["docker","exec","-e",f"RUN_ID={serverrun}","-e",f"PATCH_DIR={patch}",
         "-e",f"PORT={PORT}","-e",f"NCCL_PORT={PORT+1}","-e","HIP_VISIBLE_DEVICES="+",".join(map(str,PAIR)),
         "-e","PHYSICAL_PAIR="+",".join(map(str,PAIR)),"-e","MAX_WAIT=900","-e","MAX_GATE_WAIT=5",
         "-e","VARIANT=candidate",
         CONTAINER,"bash",str(RUNTIME/"scripts/start_service.sh")]
    command(cmd,evidence/"start-service.log",1200,"starting-service",client_concurrency=c,kind=kind)
    info=api(kind);write(evidence/"server-info.json",info)
    write(evidence/"container-inspect.json",owned())
    serverlog=ROOT/"logs"/serverrun/"server.log"
    assert serverlog.exists(),f"Expected service log {serverlog}"
    if kind=="candidate":
        text=serverlog.read_text(errors="replace")
        assert text.count("[Native BF16 AR] Installed")>=2,"Missing two-rank native guard installation"
        assert "[Native BF16 AR] Eligible rank=0" in text and "[Native BF16 AR] Eligible rank=1" in text,"Missing native eligibility"
    bench=["docker","exec","-e",f"RUN_ID={subrun}","-e",f"PORT={PORT}",
           "-e","FULL_WARMUP_ROUNDS=1","-e","MEASURE_ROUNDS=3",CONTAINER,
           "bash",str(RUNTIME/"scripts/benchmark_sharegpt.sh"),str(c)]
    command(bench,evidence/"benchmark-command.log",12*3600,"benchmarking-sharegpt",client_concurrency=c,kind=kind,result_dir=str(results.relative_to(ROOT)))
    assert results.exists()
    # Stop before freezing server logs. The host queue retains its selected pair lease.
    owned();run(["docker","stop","-t","30",CONTAINER],120)
    shutil.copy2(serverlog,evidence/"server.log")
    pidfile=serverlog.with_name("server.pid")
    assert pidfile.exists();shutil.copy2(pidfile,evidence/"server.pid")
    after=overlap()
    meta=dict(status="COMPLETE",run=subrun,server_run=serverrun,kind=kind,physical_pair=PAIR,
              client_concurrency=c,server_max_running_requests=64,cuda_graph_max_bs=64,started_utc=started,finished_utc=now(),overlap_start=before,overlap_finish=after,
              native_guard_verified=kind=="candidate",result_dir=str(results.relative_to(ROOT)),
              evidence_dir=str(evidence.relative_to(ROOT)),accuracy="NOT_RUN")
    write(OUT/"stages"/f"c{c}-{kind}.json",meta)
    ACTIVE_SERVER=None;ACTIVE_EVIDENCE=None
    state("stage-complete",client_concurrency=c,kind=kind,overlap=after)
    manifest()


def main():
    def interrupted(signum,frame):
        raise TimeoutError("Owned latest-submit queue interrupted/deadline signal "+str(signum))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,interrupted)
    assert not OUT.exists() and not STATE.exists(),"Run already exists; inspect rather than restart"
    lease=(ROOT/"result"/f"gpu{PAIR_ID}-owner.lock").open("a+")
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease.seek(0);lease.truncate();lease.write(json.dumps(dict(pid=os.getpid(),run=RUN,physical_pair=PAIR)));lease.flush()
    OUT.mkdir();(OUT/"stages").mkdir();(OUT/"source").mkdir()
    signal.alarm(36*3600)
    success=False
    try:
        obj=owned()
        gpu=json.loads(run(["/opt/dtk/bin/rocm-smi","-d","0","1","--showuse","--showmeminfo","vram","--json"]))
        assert set(gpu)=={"card0","card1"}
        for card,item in gpu.items():
            assert float(item["HCU use (%)"])==0 and int(item["vram Total Used Memory (MiB)"])<=16,(card,item)
        # Only this new task-specific container may be restarted. A stopped
        # container is valid; an already-running one must still be idle sleep.
        initial_top="Stopped owned container; no running container processes"
        if obj["State"]["Running"]:
            initial_top=run(["docker","top",CONTAINER,"-eo","pid,ppid,args"])
            assert not any(x in initial_top for x in ["sglang","bench_serving","benchmark.py","hipprof","probe_"]),"Owned container is not idle"
        runtime_manifest=json.loads((RUNTIME/"runtime-manifest.json").read_text())
        for relative,expected in runtime_manifest["files"].items():
            path=RUNTIME/relative
            assert os.path.commonpath([str(path.resolve()),str(RUNTIME.resolve())])==str(RUNTIME.resolve()),relative
            assert sha(path)==expected,relative
        shutil.copy2(RUNTIME/"runtime-manifest.json",OUT/"source/runtime-manifest.json")
        report_reference=RUNTIME/"v1-report-reference.json"
        report_source=RUNTIME/"v1-report-source.md"
        assert sha(report_reference)=="54c5ff586396dd90760eeb3fc3686f741927a5ef065294148ce293919ab4ef2d"
        assert sha(report_source)=="b36db6ec60fcac45d710fada2bdf74b1b4655ef631567e6c34fc4dc4a20e417d"
        shutil.copy2(report_reference,OUT/"source/v1-report-reference.json")
        shutil.copy2(report_source,OUT/"source/v1-report-source.md")
        for name in ["run_server64_pair01.py","review_server64_pair01.py","scripts/start_service.sh","scripts/serve.sh","scripts/benchmark_sharegpt.sh","scripts/validate_sharegpt.py"]:
            src=RUNTIME/name;assert src.is_file(),src
            dst=OUT/"source"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        patch_sha={}
        for kind in ["candidate"+PAIR_ID]:
            shutil.copytree(RUNTIME/kind,OUT/"source"/kind,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            for p in sorted((RUNTIME/kind).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    patch_sha[str(p.relative_to(ROOT))]=sha(p)
        assert patch_sha,"No latest-submit candidate artifacts"
        candidate=RUNTIME/("candidate"+PAIR_ID)
        assert sha(candidate/"sitecustomize.py")=="da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610"
        expected_guard="f9d5ec98c70724dcb80857228681a6c03c619606643cdd6918c9084af51cd7f4"
        assert sha(candidate/"native_bf16_ar_guard.py")==expected_guard
        native_manifest=json.loads((candidate/"native-ar-manifest.json").read_text())
        for rel,expected in native_manifest["files"].items():assert sha(candidate/rel)==expected,rel
        gate_path=RUNTIME/"candidate01-gate.json"
        gate=json.loads(gate_path.read_text())
        assert gate["status"]=="NUMERICAL_GRAPH_PASS" and gate["physical_pair"]=="0,1"
        assert gate["candidate_site_sha256"]==sha(candidate/"sitecustomize.py")
        assert gate["candidate_guard_sha256"]==expected_guard
        shutil.copy2(gate_path,OUT/"source/candidate01-gate.json")
        evidence_sha=sha(gate_path)
        dataset=Path("/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
        assert sha(dataset)==runtime_manifest["dataset_sha256"],"Dataset changed after runtime preparation"
        write(OUT/"preflight.json",dict(utc=now(),run=RUN,pid=os.getpid(),physical_pair=PAIR,
            candidate_sha256=patch_sha,dataset_path=str(dataset),dataset_sha256=sha(dataset),
            queue_source_sha256=sha(OUT/"source/run_server64_pair01.py"),
            reviewer_source_sha256=sha(OUT/"source/review_server64_pair01.py"),
            runtime_manifest_sha256=sha(RUNTIME/"runtime-manifest.json"),runtime_files_verified=len(runtime_manifest["files"]),
            historical_report_reference_sha256=sha(report_reference),historical_report_source_sha256=sha(report_source),
            scenes=SCENES,kinds=KINDS,server_max_running_requests=64,cuda_graph_max_bs=64,full_warmup_rounds=1,measure_rounds=3,prompts_per_round=1000,
            short_warmup_per_run=64,seed=20260912,context=67584,accuracy="NOT_RUN",overlap=overlap(),
            container=CONTAINER,port=PORT,nccl_port=PORT+1,image=IMAGE,
            numerical_gate_sha256=evidence_sha,
            user_authorization="Latest explicit user permission to run only physical HCU0,1 TP2; do not touch other users containers or HCU2,3",initial_container_running=obj["State"]["Running"],
            initial_container_processes=initial_top,scope="Latest submit candidate absolute performance only; no baseline runs or relative gain calculations"))
        for c in SCENES:
            for kind in KINDS:
                for relative,expected in patch_sha.items():
                    assert sha(ROOT/relative)==expected,relative
                stage(c,kind)
            manifest();review(f"review-c{c}",True)
        manifest();success=True
    except BaseException as exc:
        state("failed",error=repr(exc),accuracy="NOT_RUN",last_stage=CURRENT)
        shutil.copy2(STATE,OUT/"failed-state.json")
        raise
    finally:
        # This unique container is wholly owned by the matrix, including failures.
        try:
            owned();run(["docker","stop","-t","30",CONTAINER],120)
            if ACTIVE_SERVER and ACTIVE_EVIDENCE:
                for name in ["server.log","server.pid"]:
                    src=ROOT/"logs"/ACTIVE_SERVER/name
                    dst=ACTIVE_EVIDENCE/f"failure-{name}"
                    if src.exists() and not dst.exists():shutil.copy2(src,dst)
            write(OUT/"card-release.json",dict(utc=now(),container=CONTAINER,
                running=owned()["State"]["Running"],queue_success=success,physical_pair=PAIR))
            assert owned()["State"]["Running"] is False,"Owned container still running"
            write(OUT/"local-sync-policy.json",dict(max_local_file_bytes_exclusive=16*1024*1024,
                large_files_remote_only=True,manifest_explicit_files_only=True,
                note="Do not download a JSONL/log/tar/tensor/DB file >=16MiB. Keep full raw remotely; sync summary/source/SHA locally."))
            manifest()
            if success:
                review("review-final",False)
                (OUT/"finished").write_text(now()+"\n")
                state("submit-c64-server64-pair01-complete",accuracy="NOT_RUN",default_patch_changed=False,frozen=False)
                shutil.copy2(STATE,OUT/"final-state.json")
            manifest()
            control={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="control-sha256.json"}
            write(OUT/"control-sha256.json",control)
        except BaseException as exc:
            state("failed",error=repr(exc),accuracy="NOT_RUN",finalization_failure=True,last_stage=CURRENT)
            shutil.copy2(STATE,OUT/"failed-state.json")
            raise
        finally:
            signal.alarm(0)
            fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=="__main__":
    main()
