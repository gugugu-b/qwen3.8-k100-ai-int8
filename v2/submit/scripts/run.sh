#!/usr/bin/env bash
# Host entrypoint; a private container and the existing HCU6,7 task lock.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-check}
NAME=qwen38-v2-submit
IMAGE_ID=sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad
IMAGE_REF=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
PORT=${PORT:-31014}; NCCL_PORT=${NCCL_PORT:-31015}
MODEL_PATH=${MODEL_PATH:-/data/model/Qwen3.8-27B-Channel-INT8-w8a8}
DATASET_PATH=${DATASET_PATH:-/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}
LIBRARY_DIR=${LIBRARY_DIR:-$ROOT/config/library_gpu4_int8tuned_stock_clean}
# Cooperates with the other work in this workspace. Do not redirect this lock.
LOCK_FILE=/data/wangzy/submit1009/result/gpu67-owner.lock
RUN_ID=${RUN_ID:-v2-$(date -u +%Y%m%dT%H%M%SZ)-$$}
VARIANT=${VARIANT:-candidate}
case "$RUN_ID" in *[!a-zA-Z0-9_.-]*|'') echo 'Invalid RUN_ID' >&2; exit 2;; esac
case "$VARIANT" in candidate) PATCH_DIR="$ROOT/patch";; baseline-v1) PATCH_DIR="$ROOT/patch/baseline-v1";; *) exit 2;; esac
verify_container() {
 docker inspect "$NAME" | python3 -I -c 'import json,sys; x=json.load(sys.stdin)[0]; assert (x["Config"].get("Labels") or {}).get("qwen38.v2.submit")=="20261016"; assert x["Image"]=="sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"; assert "HIP_VISIBLE_DEVICES=6,7" in x["Config"]["Env"]'
}
case "$ACTION" in
 check) python3 -I "$ROOT/scripts/verify_package.py";;
 check-runtime)
  python3 -I "$ROOT/scripts/verify_package.py" --runtime --library "$LIBRARY_DIR"
  test -d "$MODEL_PATH"; docker image inspect "$IMAGE_ID" >/dev/null
  ;;
 start)
  python3 -I "$ROOT/scripts/verify_package.py" --runtime --library "$LIBRARY_DIR"
  test -d "$MODEL_PATH"; test -d /opt/hyhal
  if ! docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then docker pull "$IMAGE_REF"; fi
  test "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" = "$IMAGE_ID" || { echo 'Image ID mismatch' >&2; exit 2; }
  if docker inspect "$NAME" >/dev/null 2>&1; then echo "$NAME exists; inspect or stop it first" >&2; exit 2; fi
  mkdir -p "$(dirname "$LOCK_FILE")"; touch "$LOCK_FILE"
  mounts=()
  for config in "$ROOT"/config/aiter/*/*.json; do
   stem=$(basename "$(dirname "$config")")
   mounts+=(-v "$config:/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs/$stem/$(basename "$config"):ro")
  done
  if [[ -f "$DATASET_PATH" ]]; then mounts+=(-v "$DATASET_PATH:$DATASET_PATH:ro"); fi
  docker run -d --name "$NAME" --label qwen38.v2.submit=20261016 \
   --network host --ipc host --device /dev/kfd --device /dev/mkfd --device /dev/dri \
   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
   -e HIP_VISIBLE_DEVICES=6,7 -e PORT="$PORT" -e NCCL_PORT="$NCCL_PORT" \
   -e MODEL_PATH="$MODEL_PATH" -e DATASET_PATH="$DATASET_PATH" -e PATCH_DIR="$PATCH_DIR" \
   -e VARIANT="$VARIANT" -e RUN_ID="$RUN_ID" -e MAX_WAIT=1800 -e MAX_GATE_WAIT=5 \
   -e TVM_FFI_CACHE_DIR=/run/qwen38-tvm-ffi \
   -v "$ROOT:$ROOT" -v "$MODEL_PATH:$MODEL_PATH:ro" -v /opt/hyhal:/opt/hyhal:ro \
   -v "$LIBRARY_DIR:/run/qwen38-library:ro" -v "$LOCK_FILE:/run/qwen38-gpu-pair.lock" \
   -v "$ROOT/patch/runtime/tvm-ffi:/run/qwen38-tvm-ffi:ro" \
   -v "$ROOT/patch/runtime/direct_register_custom_op.py:/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py:ro" \
   "${mounts[@]}" "$IMAGE_ID" sleep infinity
  if ! docker exec "$NAME" bash "$ROOT/scripts/start_server.sh"; then
   docker logs "$NAME" >&2; docker stop --time 20 "$NAME"; docker rm "$NAME"; exit 1
  fi
  echo "Service: http://127.0.0.1:$PORT ; stop: bash scripts/run.sh stop"
  ;;
 benchmark)
  verify_container
  docker exec -e RUN_ID="$RUN_ID" "$NAME" bash "$ROOT/scripts/benchmark_reference.sh" "${2:-4k}" "${3:-8}"
  ;;
 sharegpt)
  verify_container
  docker exec -e RUN_ID="$RUN_ID" "$NAME" bash "$ROOT/scripts/benchmark_sharegpt.sh" "${2:-32}"
  ;;
 demo)
  verify_container; docker exec "$NAME" python3 "$ROOT/scripts/demo_client.py" "${2:-请用中文简要介绍你的能力。}"
  ;;
 stop)
  if docker inspect "$NAME" >/dev/null 2>&1; then verify_container; docker stop --time 20 "$NAME"; docker rm "$NAME"; fi
  ;;
 *) echo 'Usage: run.sh {check|check-runtime|start|benchmark [4k|64k] [C]|sharegpt [C]|demo [prompt]|stop}' >&2; exit 2;;
esac
