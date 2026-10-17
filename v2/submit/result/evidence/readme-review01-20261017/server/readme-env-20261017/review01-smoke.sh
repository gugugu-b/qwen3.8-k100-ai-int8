#!/usr/bin/env bash
set -euo pipefail
cd /data/wangzy/v2/submit
export SUBMIT_ROOT="$PWD"
export REVIEW_ID="review01-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export MODEL_PATH=/data/model/Qwen3.8-27B-Channel-INT8-w8a8
export DATASET_PATH=/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
export LIBRARY_DIR="$SUBMIT_ROOT/config/library_gpu4_int8tuned_stock_clean"
export PORT=31014
export NCCL_PORT=31015
export LOCK_FILE=/data/wangzy/submit1009/result/gpu01-owner.lock
export PATCH_DIR="$SUBMIT_ROOT/result/runs/${REVIEW_ID}-runtime/patch"
export REVIEW_IMAGE_ID=sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad
OUT=/data/wangzy/v2/readme-env-20261017
mkdir -p "$OUT"
printf "%s\n" "$REVIEW_ID" > "$OUT/review01-id.txt"
if docker inspect qwen38-v2-review01 >/dev/null 2>&1; then echo "Review container already exists"; exit 2; fi
command -v docker python3 flock
cat /sys/module/hycu/version
test -d /opt/hyhal && test -e /dev/kfd && test -e /dev/mkfd && test -d /dev/dri
/opt/dtk/bin/rocm-smi -d 0 1 --showuse --showmeminfo vram
# 确认0、1没有其他计算任务；下面检查协作锁及两个端口。
mkdir -p "$(dirname "$LOCK_FILE")"
touch "$LOCK_FILE"
flock -n "$LOCK_FILE" true
python3 -I - "$PORT" "$NCCL_PORT" <<'CHECK_PORTS'
import socket,sys
for port in map(int,sys.argv[1:]):
    with socket.socket() as s:
        s.bind(('0.0.0.0',port))
print('PORTS_FREE')
CHECK_PORTS
bash scripts/run_review01.sh check
python3 -I scripts/verify_patch_bundle.py
python3 -I scripts/verify_package.py --runtime --library "$LIBRARY_DIR"
mkdir -p "result/runs/${REVIEW_ID}-assets"
python3 -I scripts/check_assets.py \
  --model-path "$MODEL_PATH" --dataset-path "$DATASET_PATH" \
  --output "result/runs/${REVIEW_ID}-assets/assets.json"
python3 -I scripts/prepare_review01.py --output "$PATCH_DIR"
cleanup() { bash scripts/run_review01.sh stop; }; trap cleanup EXIT
mounts=()
for config in "$SUBMIT_ROOT"/config/aiter/*/*.json; do
  stem=$(basename "$(dirname "$config")")
  mounts+=(-v "$config:/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs/$stem/$(basename "$config"):ro")
done

docker run -d --name qwen38-v2-review01 --label qwen38.v2.review01=20261017 \
  --network host --ipc host \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES=0,1 -e GPU_PAIR=0,1 \
  -e PORT="$PORT" -e NCCL_PORT="$NCCL_PORT" \
  -e MODEL_PATH="$MODEL_PATH" -e DATASET_PATH="$DATASET_PATH" \
  -e PATCH_DIR="$PATCH_DIR" -e VARIANT=candidate \
  -e RUN_ID="${REVIEW_ID}-server" -e MAX_WAIT=1800 -e MAX_GATE_WAIT=5 \
  -e TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi \
  -v "$SUBMIT_ROOT:$SUBMIT_ROOT" \
  -v "$MODEL_PATH:$MODEL_PATH:ro" -v "$DATASET_PATH:$DATASET_PATH:ro" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$LIBRARY_DIR:/run/qwen38-library:ro" \
  -v "$LOCK_FILE:/run/qwen38-gpu-pair.lock" \
  -v "$SUBMIT_ROOT/patch/runtime/tvm-ffi:/run/qwen38-tvm-ffi:ro" \
  -v "$SUBMIT_ROOT/patch/runtime/direct_register_custom_op.py:/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py:ro" \
  "${mounts[@]}" "$REVIEW_IMAGE_ID" sleep infinity
docker exec -i qwen38-v2-review01 bash -s <<'CHECK_RUNTIME'
set -euo pipefail
source /opt/dtk/env.sh
python3 -I - <<'PY_CHECK'
import hashlib,importlib.metadata as m,json,os
from pathlib import Path
import torch
for name in ('sglang','torch','triton','aiter','transformers','aiohttp','requests'):
    print(name, m.version(name))
assert os.environ['HIP_VISIBLE_DEVICES'] == os.environ['GPU_PAIR'] == '0,1'
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
for i in range(2):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).gcnArchName)
p = Path(os.environ['PATCH_DIR'])
manifest = json.loads((p/'native-ar-manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == digest, name
assert "assert os.environ['HIP_VISIBLE_DEVICES'] == '0,1'" in (p/'native_bf16_ar_guard.py').read_text()
print('REVIEW01_RUNTIME_PASS')
PY_CHECK
CHECK_RUNTIME
bash scripts/run_review01.sh check-runtime
RUN_ID="${REVIEW_ID}-server" bash scripts/run_review01.sh start
bash scripts/run_review01.sh demo "请用一句中文解释张量并行。"
docker inspect qwen38-v2-review01 > "$OUT/review01-container.json"
printf "REVIEW01_README_SMOKE_PASS %s\n" "$REVIEW_ID"
