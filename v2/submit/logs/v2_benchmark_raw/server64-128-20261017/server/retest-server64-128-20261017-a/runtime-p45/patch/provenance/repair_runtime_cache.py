#!/usr/bin/env python3
"""Preserve the failed fresh-image evidence and add the measured runtime cache."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path('/data/wangzy/submit1009')
PACKAGE = ROOT / 'delivery/20261010-4k1k/submit1009'
EVIDENCE = ROOT / 'result/delivery-package-runtime-repair-20261010-b'
EVIDENCE.mkdir(exist_ok=False)
failed = json.loads((ROOT / 'result/delivery-package-startup-20261010-a/review.json').read_text())
assert failed['status'] == 'FAILED' and failed['gpu67_lock'] == 'FREE'
for name in ('scripts/run.sh', 'README.md', 'env_declare.yaml', 'config/package-sha256.json'):
    target = EVIDENCE / 'before' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PACKAGE / name, target)
cache = ROOT / 'result/delivery-package-runtime-cache-20261010-a/libtorch_c_dlpack_addon_torch210-rocm.so'
target = PACKAGE / 'patch/runtime/tvm-ffi' / cache.name
target.parent.mkdir(exist_ok=False)
shutil.copy2(cache, target)
digest = hashlib.sha256(cache.read_bytes()).hexdigest()
run = PACKAGE / 'scripts/run.sh'
text = run.read_text()
needle = '   -e MAX_WAIT=600 -e MAX_GATE_WAIT=5 \\\n'
assert text.count(needle) == 1
text = text.replace(needle, needle + '   -e TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi \\\n   -v "$PACKAGE_DIR/patch/runtime/tvm-ffi:/run/qwen38-tvm-ffi:ro" \\\n')
run.write_text(text)
env_path = PACKAGE / 'env_declare.yaml'
env = json.loads(env_path.read_text())
env['runtime_cache'] = dict(source='Exact cache copied from the stopped, measured qwen38-submit1009-diag container',
    file=str(target.relative_to(PACKAGE)), sha256=digest, bytes=target.stat().st_size,
    mount='/run/qwen38-tvm-ffi:ro', variable='TVM_FFI_CACHE_DIR',
    reason='Fresh image has no cached Torch C DLPack addon; Python site import recurses into the optional builder',
    no_tvm_disable=True, shared_installation_modified=False)
env_path.write_text(json.dumps(env, indent=2) + '\n')
readme = PACKAGE / 'README.md'
readme.write_text(readme.read_text() + '\n新容器运行依赖：`patch/runtime/tvm-ffi/` 含本次实测容器已有的Torch C DLPack缓存（原文件200624字节，SHA见env_declare.yaml）。入口以只读目录挂载并指定TVM_FFI_CACHE_DIR，保留已测运行依赖，不关闭TVM功能、不修改共享库。首次新镜像检查的递归构建失败完整保留，修正后使用独立RUN检查。\n')
manifest = {str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p != PACKAGE / 'config/package-sha256.json'}
(PACKAGE / 'config/package-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
review = dict(status='MEASURED_RUNTIME_CACHE_PACKAGED', cache_sha256=digest,
    cache_bytes=target.stat().st_size, source_container='qwen38-submit1009-diag',
    unchanged_candidate_sha256=hashlib.sha256((PACKAGE / 'patch/sitecustomize.py').read_bytes()).hexdigest(),
    previous_smoke='FAILED_RECURSIVE_OPTIONAL_BUILDER', current_smoke='NOT_RUN',
    performance_data_modified=False, shared_library_modified=False)
(EVIDENCE / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review))
