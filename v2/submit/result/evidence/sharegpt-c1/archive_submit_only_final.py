#!/usr/bin/env python3
"""One-time CPU archive of a finished latest-submit performance queue.

Run on the server only. This never starts/stops a container or writes the
original queue, sources, manifests, or lock contents. All new files live in a
unique sibling final-postcheck directory. Failed new archives remain evidence.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT_DEFAULT = Path("/data/wangzy/submit1009")
IMAGE = "sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
LABEL = "qwen38.submit1009.performance"
FINAL_PHASE = "submit-candidate-performance-complete"
REVIEW_PASS = "LATEST_SUBMIT_ABSOLUTE_PERFORMANCE_RAW_REVIEW_PASS"
LIMIT = 16 * 1024 * 1024
MUTABLE = {"artifact-sha256.json", "performance-summary.json", "finished",
           "final-state.json", "failed-state.json", "control-sha256.json"}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def path_in(root, relative):
    require(isinstance(relative, str) and relative, "Empty/non-string manifest key")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == relative,
            "Invalid relative manifest path: " + relative)
    path = root / rel
    require(os.path.commonpath([str(path.resolve()), str(root)]) == str(root),
            "Manifest path escapes root: " + relative)
    require(path.is_file(), "Missing manifest file: " + relative)
    return path


def verify_manifest(root, manifest):
    require(isinstance(manifest, dict) and manifest, "Empty/non-object manifest")
    for relative, digest in manifest.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Invalid SHA256: " + relative)
        require(sha(path_in(root, relative)) == digest, "SHA256 mismatch: " + relative)


def inspect_owned(root, container, pair):
    proc = subprocess.run(["docker", "inspect", container], check=True,
                          capture_output=True, text=True, timeout=30)
    items = json.loads(proc.stdout)
    require(len(items) == 1, "Unexpected owned-container inspection result")
    obj = items[0]
    require(obj["Name"].lstrip("/") == container, "Owned container name differs")
    require(obj["Image"] == IMAGE, "Owned container image differs")
    require(obj["Config"].get("Labels", {}).get(LABEL) == "20261016", "Owned label differs")
    require("HIP_VISIBLE_DEVICES=" + ",".join(map(str, pair)) in obj["Config"]["Env"],
            "Owned physical pair differs")
    require(any(m["Source"] == str(root) and m["Destination"] == str(root)
                for m in obj["Mounts"]), "Owned ROOT mount missing")
    require(obj["State"]["Running"] is False and obj["State"]["Pid"] == 0,
            "Owned container has not stopped")
    return dict(utc=now(),container=container,id=obj["Id"],image=obj["Image"],
                state=obj["State"],physical_pair=pair,label={LABEL:"20261016"},
                root_mount_verified=True)


def require_absent(pid):
    require(isinstance(pid, int) and pid > 1, "Invalid queue PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    raise AssertionError("Queue PID still exists (including possible reuse): " + str(pid))


def expected_artifact_paths(root, out, run, scenes):
    paths = {p for p in out.rglob("*") if p.is_file() and p.name not in MUTABLE
             and not any(x.startswith("review-") for x in p.relative_to(out).parts)}
    for c in scenes:
        stage = json.loads((out / "stages" / f"c{c}-candidate.json").read_text())
        directory = root / stage["result_dir"]
        require(directory.is_dir(), "Missing completed-stage raw directory")
        paths.update(p for p in directory.rglob("*") if p.is_file())
        raw = root / "experiments/performance-validation-20261016/result" / f"{run}-c{c}-candidate" / f"sharegpt-c{c}"
        if raw.exists():
            paths.update(p for p in raw.rglob("*") if p.is_file())
    return {str(p.relative_to(root)) for p in paths}


def copy_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Refusing archive overwrite: " + str(target))
    require(sha(source) == expected, "Source changed before copy: " + str(source))
    shutil.copy2(source, target)
    require(sha(target) == expected and sha(source) == expected,
            "Copy or source changed: " + str(source))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair45", action="store_true")
    mode.add_argument("--pair67", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    a = p.parse_args()
    root = a.root.resolve()
    pair_id = "45" if a.pair45 else "67"
    pair = [4, 5] if a.pair45 else [6, 7]
    scenes = [1] if a.pair45 else [16, 64, 4]
    mode_flag = "--pair" + pair_id
    run = "submit-current-c1-pair45-20261016-a" if a.pair45 else "submit-current-sharegpt-pair67-20261016-a"
    out = root / "result" / run
    post = root / "result" / (run + "-final-postcheck")
    state_path = root / "result" / (run + "-state.json")
    job_path = root / "logs" / (run + "-job.log")
    lock_path = root / "result" / ("gpu" + pair_id + "-owner.lock")
    container = "qwen38-submit1009-perf" + pair_id + "-20261016"
    require(not post.exists(), "Never overwrite or retry an existing final archive")
    require(out.is_dir() and lock_path.is_file(), "Queue or existing pair lock missing")
    require(state_path.is_file() and job_path.is_file(), "External terminal state/job missing")
    phase = "readiness-before-mkdir"
    created = False
    with lock_path.open("rb") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_bytes = lease.read()
        try:
            state_bytes = state_path.read_bytes()
            state = json.loads(state_bytes)
            require(state["phase"] == FINAL_PHASE and state["run"] == run
                    and state["physical_pair"] == pair, "Queue is not finally complete")
            require(state["accuracy"] == "NOT_RUN" and state["default_patch_changed"] is False
                    and state["frozen"] is False, "Unexpected final scope")
            require_absent(state["pid"])
            required = ["finished", "final-state.json", "card-release.json", "local-sync-policy.json",
                        "artifact-sha256.json", "control-sha256.json", "performance-summary.json",
                        "review-final/performance-summary.json", "review-final/review-sha256.json"]
            require(all((out / name).is_file() for name in required), "Final control files missing")
            require((out / "finished").read_text().strip(), "Empty finished marker")
            require((out / "final-state.json").read_bytes() == state_bytes,
                    "External terminal state differs from frozen final-state")
            require(not (out / "failed-state.json").exists(), "Original finalization failed")
            preflight = json.loads((out / "preflight.json").read_text())
            require(preflight["pid"] == state["pid"] and preflight["run"] == run
                    and preflight["physical_pair"] == pair, "Final PID/preflight identity differs")
            release = json.loads((out / "card-release.json").read_text())
            require(release["queue_success"] is True and release["running"] is False
                    and release["container"] == container and release["physical_pair"] == pair,
                    "Successful card-release proof missing")
            container_before = inspect_owned(root, container, pair)
            require({x.name for x in (out / "stages").glob("*.json")}
                    == {f"c{c}-candidate.json" for c in scenes}, "Final scene set differs")
            for c in scenes:
                stage = json.loads((out / "stages" / f"c{c}-candidate.json").read_text())
                require(stage["status"] == "COMPLETE" and stage["kind"] == "candidate"
                        and stage["physical_pair"] == pair and stage["client_concurrency"] == c,
                        "Incomplete/unexpected candidate stage")
            artifact_bytes = (out / "artifact-sha256.json").read_bytes()
            control_bytes = (out / "control-sha256.json").read_bytes()
            artifact, control = json.loads(artifact_bytes), json.loads(control_bytes)
            verify_manifest(root, artifact)
            verify_manifest(root, control)
            require(set(artifact) == expected_artifact_paths(root, out, run, scenes),
                    "Artifact manifest does not exactly cover final source/evidence/raw")
            expected_control = {str(x.relative_to(root)) for x in out.rglob("*")
                                if x.is_file() and x.name != "control-sha256.json"}
            require(set(control) == expected_control, "Final control manifest coverage differs")
            producer_bytes = (out / "review-final/performance-summary.json").read_bytes()
            require(producer_bytes == (out / "performance-summary.json").read_bytes(),
                    "Producer final summary bytes differ")
            producer = json.loads(producer_bytes)
            require(producer["status"] == REVIEW_PASS and producer["complete"] is True
                    and producer["physical_pair"] == pair and producer["run"] == run
                    and set(producer["scenarios"]) == set(map(str, scenes)), "Producer final review incomplete")
            require(producer["manifest_sha256"] == sha(out / "artifact-sha256.json"),
                    "Producer final review used another artifact manifest")
            union = dict(artifact)
            for relative, digest in control.items():
                require(relative not in union or union[relative] == digest, "Conflicting manifest SHA")
                union[relative] = digest
            for name in ["artifact-sha256.json", "control-sha256.json"]:
                relative = str((out / name).relative_to(root))
                union[relative] = sha(out / name)
            job_sha = sha(job_path)
            state_sha = sha(state_path)
            # All readiness and exact source checks pass before any archive exists.
            post.mkdir()
            created = True
            phase = "copying-final-source-and-control-union"
            snap = post / "submit1009"
            for relative, digest in sorted(union.items()):
                copy_verified(path_in(root, relative), snap / relative, digest)
            diagnostics = post / "control"
            diagnostics.mkdir()
            copy_verified(state_path, diagnostics / "current-queue-state.json", state_sha)
            copy_verified(job_path, diagnostics / "queue-job.log", job_sha)
            save(diagnostics / "container-before.json", container_before)
            save(diagnostics / "pair-lock.json", dict(utc=now(), path=str(lock_path), physical_pair=pair,
                 nonblocking_lock_was_free=True, cpu_archive_lease_held=True,
                 existing_lock_contents_sha256=hashlib.sha256(lock_bytes).hexdigest(), contents_changed=False))
            save(diagnostics / "readiness.json", dict(utc=now(), run=run, physical_pair=pair,
                 original_queue_pid=state["pid"], queue_pid_absent=True, final_phase=FINAL_PHASE,
                 artifact_count=len(artifact), control_count=len(control), union_files=len(union),
                 artifact_keys_relative_to="ROOT", control_keys_relative_to="ROOT",
                 artifact_exact_coverage=True, control_exact_coverage=True, source_manifests_unchanged=True))
            copy_verified(Path(__file__).resolve(), post / "archive_submit_only_final.py", sha(Path(__file__).resolve()))
            phase = "strict-frozen-final-independent-review"
            reviewer = snap / "result" / run / "source/review_submit_only.py"
            independent = post / "remote-independent-review"
            require(not independent.exists(), "Independent review output already exists")
            with (post / "independent-review.log").open("x") as log:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                subprocess.run([sys.executable, "-B", str(reviewer), mode_flag,
                     "--root", str(snap), "--run", run, "--out", str(independent)],
                     check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
            independent_bytes = (independent / "performance-summary.json").read_bytes()
            require(independent_bytes == producer_bytes, "Final independent/producer summary bytes differ")
            reviewed = json.loads(independent_bytes)
            require(reviewed["complete"] is True and set(reviewed["scenarios"]) == set(map(str, scenes)),
                    "Strict independent review scene set incomplete")
            phase = "final-immutability-and-release-check"
            verify_manifest(root, artifact)
            verify_manifest(root, control)
            require((out / "artifact-sha256.json").read_bytes() == artifact_bytes
                    and (out / "control-sha256.json").read_bytes() == control_bytes,
                    "Original final manifest bytes changed")
            require(state_path.read_bytes() == state_bytes and sha(job_path) == job_sha,
                    "Terminal state/job changed during archival")
            require_absent(state["pid"])
            save(diagnostics / "container-after.json", inspect_owned(root, container, pair))
            lease.seek(0)
            require(lease.read() == lock_bytes, "Pair lock contents changed during archive")
            proof = dict(status="LATEST_SUBMIT_FINAL_ARCHIVE_INDEPENDENT_REVIEW_PASS", utc=now(),
                 run=run, physical_pair=pair, scenes=scenes, complete=True,
                 queue_pid_absent=True, owned_container_stopped=True, nonblocking_pair_lock_verified_free=True,
                 artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                 control_sha256=hashlib.sha256(control_bytes).hexdigest(),
                 producer_summary_sha256=hashlib.sha256(producer_bytes).hexdigest(),
                 independent_summary_sha256=hashlib.sha256(independent_bytes).hexdigest(),
                 summaries_byte_identical=True, source_union_files=len(union),
                 original_queue_sources_results_manifests_unchanged=True,
                 container_mutated=False, lock_contents_changed=False,
                 accuracy="NOT_RUN", formal_acceptance=False, default_patch_changed=False, frozen=False,
                 caveat="Historical v1 report comparison only. No new v1 run, accuracy test, promotion or freeze.")
            save(post / "archive-review.json", proof)
            files = {str(x.relative_to(post)):dict(bytes=x.stat().st_size, sha256=sha(x))
                     for x in sorted(post.rglob("*")) if x.is_file()}
            save(post / "archive-inventory.json", dict(status="LATEST_SUBMIT_FINAL_ARCHIVE_COMPLETE",
                 utc=now(), run=run, physical_pair=pair, scenes=scenes, files=files,
                 inventory_keys_relative_to="POST", inventory_excludes_itself=True,
                 large_remote_only=[key for key,value in files.items() if value["bytes"] >= LIMIT],
                 local_download_policy="Explicit inventory files smaller than 16MiB only; retain every larger file remotely",
                 snapshot_proof=proof))
            print(json.dumps(dict(status="LATEST_SUBMIT_FINAL_ARCHIVE_COMPLETE", run=run,
                 physical_pair=pair, post=str(post), files=len(files), complete=True,
                 summary_sha256=proof["independent_summary_sha256"],
                 large_remote_only=[key for key,value in files.items() if value["bytes"] >= LIMIT])))
        except BaseException as exc:
            if created:
                failure = post / "archive-failure.json"
                if not failure.exists():
                    save(failure, dict(status="FINAL_ARCHIVE_FAILED_RETAINED", utc=now(),
                         run=run, physical_pair=pair, phase=phase, error=repr(exc),
                         traceback=traceback.format_exc(), original_queue_modified=False,
                         note="Keep this unique failed archive intact; never overwrite or delete it to retry."))
            raise
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
