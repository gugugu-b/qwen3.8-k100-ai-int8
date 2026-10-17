#!/usr/bin/env python3
"""Exclusive HCU6,7 ShareGPT performance-only matrix, no accuracy/promotion."""
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
RUNTIME = ROOT/"experiments/performance-validation-20261016"
RUN = "perf-sharegpt-20261016-a"
CONTAINER = "qwen38-submit1009-perf67-20261016"
PORT = 31014
STATE = ROOT/"result"/f"{RUN}-state.json"
OUT = ROOT/"result"/RUN
IMAGE = "sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
SCENES = [32,16,64,4,1]
KINDS = ["v1","candidate","v1-return"]
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
    CURRENT=dict(phase=phase,pid=os.getpid(),utc=now(),run=RUN,physical_pair=[6,7],
                 elapsed_s=time.monotonic()-START,**kw)
    tmp=STATE.with_suffix(".tmp")
    write(tmp,CURRENT);tmp.replace(STATE)
    print(json.dumps(CURRENT),flush=True)


def run(args,timeout=180):
    return subprocess.run(args,check=True,timeout=timeout,text=True,capture_output=True).stdout


def owned():
    o=json.loads(run(["docker","inspect",CONTAINER]))[0]
    assert o["Image"]==IMAGE
    assert "HIP_VISIBLE_DEVICES=6,7" in o["Config"]["Env"]
    assert any(x["Source"]==str(ROOT) and x["Destination"]==str(ROOT) for x in o["Mounts"])
    return o


def overlap():
    result={"utc":now()}
    for name in ["perf-64k-20261016-a-state.json","perf-fixed-20261016-a-state.json","perf-other-20261016-a-state.json","perf-fixed45-20261016-a-state.json"]:
        p=ROOT/"result"/name
        if p.exists():
            result[name]=json.loads(p.read_text())
    for name in ["qwen38-submit1009-perf45-20261016"]:
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
    deadline=time.monotonic()+900
    while True:
        try:
            assert get("health")==200
            info=get("get_server_info")
            args=info.get("server_args",info)
            expected={"tp_size":2,"pp_size":1,"context_length":67584,
                "mem_fraction_static":.85,"max_running_requests":32,"max_prefill_tokens":65536,
                "chunked_prefill_size":8192,"mamba_scheduler_strategy":"no_buffer",
                "cuda_graph_max_bs":32,"disable_cuda_graph":False,"disable_radix_cache":True,
                "attention_backend":"fa3","mm_attention_backend":"fa3","page_size":64,
                "speculative_num_steps":1,"speculative_eagle_topk":1,"speculative_num_draft_tokens":2}
            for key,value in expected.items():assert args.get(key)==value,(key,args.get(key),value)
            assert args["speculative_algorithm"] in ["NEXTN","EAGLE"]
            if kind=="candidate":
                assert args["custom_all_reduce_backend"]=="native" and args["disable_custom_all_reduce"] is False
            else:
                assert args["disable_custom_all_reduce"] is True
                assert args.get("custom_all_reduce_backend") in ["off",None]
            return info
        except (OSError,ValueError):
            assert time.monotonic()<deadline,"Service readiness timeout"
            time.sleep(10)


def manifest():
    mutable={"artifact-sha256.json","comparison.json","finished","final-state.json","failed-state.json","control-sha256.json"}
    paths=[p for p in OUT.rglob("*") if p.is_file() and p.name not in mutable and not any(x.startswith("review-") for x in p.relative_to(OUT).parts)]
    for stagefile in sorted((OUT/"stages").glob("*.json")):
        stage=json.loads(stagefile.read_text())
        paths.extend(p for p in (ROOT/stage["result_dir"]).rglob("*") if p.is_file())
    values={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    write(OUT/"artifact-sha256.json",values)


def review(name,partial):
    args=["python3",str(RUNTIME/"review_sharegpt_matrix.py"),"--root",str(ROOT),"--run",RUN,"--out",str(OUT/name)]
    if partial:args.append("--allow-partial")
    command(args,OUT/f"{name}.log",3600,"cpu-review",review=name)
    comparison=json.loads((OUT/name/"comparison.json").read_text())
    write(OUT/"comparison.json",comparison)


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
    patch=RUNTIME/("candidate67" if kind=="candidate" else "baseline-v1")
    cmd=["docker","exec","-e",f"RUN_ID={serverrun}","-e",f"PATCH_DIR={patch}",
         "-e",f"PORT={PORT}","-e","NCCL_PORT=31015","-e","HIP_VISIBLE_DEVICES=6,7",
         "-e","PHYSICAL_PAIR=6,7","-e","MAX_WAIT=900","-e","MAX_GATE_WAIT=5",
         "-e",f"VARIANT={'candidate' if kind=='candidate' else 'baseline-v1'}",
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
    # Stop before freezing server logs. The host queue continues owning gpu67.
    owned();run(["docker","stop","-t","30",CONTAINER],120)
    shutil.copy2(serverlog,evidence/"server.log")
    pidfile=serverlog.with_name("server.pid")
    assert pidfile.exists();shutil.copy2(pidfile,evidence/"server.pid")
    after=overlap()
    meta=dict(status="COMPLETE",run=subrun,server_run=serverrun,kind=kind,physical_pair=[6,7],
              client_concurrency=c,started_utc=started,finished_utc=now(),overlap_start=before,overlap_finish=after,
              native_guard_verified=kind=="candidate",result_dir=str(results.relative_to(ROOT)),
              evidence_dir=str(evidence.relative_to(ROOT)),accuracy="NOT_RUN")
    write(OUT/"stages"/f"c{c}-{kind}.json",meta)
    ACTIVE_SERVER=None;ACTIVE_EVIDENCE=None
    state("stage-complete",client_concurrency=c,kind=kind,overlap=after)
    manifest()


def main():
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(SystemExit("SIGTERM requested")))
    assert not OUT.exists() and not STATE.exists(),"Run already exists; inspect rather than restart"
    lease=(ROOT/"result/gpu67-owner.lock").open("a+")
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease.seek(0);lease.truncate();lease.write(json.dumps(dict(pid=os.getpid(),run=RUN,physical_pair=[6,7])));lease.flush()
    OUT.mkdir();(OUT/"stages").mkdir();(OUT/"source").mkdir()
    success=False
    try:
        obj=owned()
        assert obj["State"]["Running"] is True
        initial_top=run(["docker","top",CONTAINER,"-eo","pid,ppid,args"])
        assert not any(x in initial_top for x in ["sglang","bench_serving","benchmark.py","hipprof","probe_"]),"Owned container is not idle"
        runtime_manifest=json.loads((RUNTIME/"runtime-manifest.json").read_text())
        for relative,expected in runtime_manifest["files"].items():
            path=RUNTIME/relative
            assert os.path.commonpath([str(path.resolve()),str(RUNTIME.resolve())])==str(RUNTIME.resolve()),relative
            assert sha(path)==expected,relative
        shutil.copy2(RUNTIME/"runtime-manifest.json",OUT/"source/runtime-manifest.json")
        for name in ["sharegpt_queue.py","review_sharegpt_matrix.py","scripts/start_service.sh","scripts/serve.sh","scripts/benchmark_sharegpt.sh","scripts/validate_sharegpt.py"]:
            src=RUNTIME/name;assert src.is_file(),src
            dst=OUT/"source"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        patch_sha={}
        for kind in ["candidate67","baseline-v1"]:
            shutil.copytree(RUNTIME/kind,OUT/"source"/kind,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            for p in sorted((RUNTIME/kind).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    patch_sha[str(p.relative_to(ROOT))]=sha(p)
        assert patch_sha,"No candidate/baseline artifacts"
        assert sha(RUNTIME/"candidate67/sitecustomize.py")=="da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610"
        assert sha(RUNTIME/"candidate67/native_bf16_ar_guard.py")=="55485f3994417b012fd76f3cf423b03b33fd90d52e2c73d98d0fc9315ab0b5aa"
        assert sha(RUNTIME/"baseline-v1/sitecustomize.py")=="f1789532d58daa66187ba901d3d7513ff142cb33fafc7793ce44f4e821464495"
        dataset=Path("/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
        assert sha(dataset)==runtime_manifest["dataset_sha256"],"Dataset changed after runtime preparation"
        write(OUT/"preflight.json",dict(utc=now(),run=RUN,pid=os.getpid(),physical_pair=[6,7],
            candidate_and_baseline_sha256=patch_sha,dataset_path=str(dataset),dataset_sha256=sha(dataset),
            scenes=SCENES,kinds=KINDS,full_warmup_rounds=1,measure_rounds=3,prompts_per_round=1000,
            short_warmup_per_run=64,seed=20260912,context=67584,accuracy="NOT_RUN",overlap=overlap()))
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
                running=owned()["State"]["Running"],queue_success=success,physical_pair=[6,7]))
            assert owned()["State"]["Running"] is False,"Owned container still running"
            write(OUT/"local-sync-policy.json",dict(max_local_file_bytes_exclusive=16*1024*1024,
                large_files_remote_only=True,manifest_explicit_files_only=True,
                note="Do not download a JSONL/log/tar/tensor/DB file >=16MiB. Keep full raw remotely; sync summary/source/SHA locally."))
            manifest()
            if success:
                review("review-final",False)
                (OUT/"finished").write_text(now()+"\n")
                state("sharegpt-matrix-complete",accuracy="NOT_RUN",default_patch_changed=False,frozen=False)
                shutil.copy2(STATE,OUT/"final-state.json")
            manifest()
            control={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="control-sha256.json"}
            write(OUT/"control-sha256.json",control)
        finally:
            fcntl.flock(lease,fcntl.LOCK_UN);lease.close()


if __name__=="__main__":
    main()
