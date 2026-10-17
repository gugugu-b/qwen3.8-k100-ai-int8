#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-start}
NAME=qwen38-submit1009-delivery
IMAGE=sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad
PORT=${PORT:-31014}
NCCL_PORT=${NCCL_PORT:-31015}
MODEL_PATH=${MODEL_PATH:-/data/model/Qwen3.8-27B-Channel-INT8-w8a8}
RUN_ID=${RUN_ID:-delivery-$(date -u +%Y%m%dT%H%M%SZ)-$$}
VARIANT=${VARIANT:-candidate}
GPU_LOCK_FILE=${GPU_LOCK_FILE:-/data/wangzy/submit1009/result/gpu67-owner.lock}
case "$VARIANT" in candidate|baseline-v1) ;; *) echo 'VARIANT must be candidate or baseline-v1' >&2; exit 2;; esac
if [[ "$VARIANT" == candidate ]]; then PATCH_DIR="$PACKAGE_DIR/patch"; else PATCH_DIR="$PACKAGE_DIR/patch/baseline-v1"; fi

verify_container() {
 docker inspect "$NAME" | python3 -I -c 'import json,sys; x=json.load(sys.stdin)[0]; assert x["Config"]["Labels"].get("qwen38.submit1009.delivery")=="20261010"; assert x["Image"]=="sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"; assert "HIP_VISIBLE_DEVICES=6,7" in x["Config"]["Env"]'
}
case "$ACTION" in
 check)
  python3 -I "$PACKAGE_DIR/scripts/verify_package.py"
  ;;
 start)
  python3 -I "$PACKAGE_DIR/scripts/verify_package.py"
  if docker inspect "$NAME" >/dev/null 2>&1; then echo "Container $NAME already exists; inspect it or use run.sh stop" >&2; exit 2; fi
  test -d "$MODEL_PATH"
  mkdir -p "$(dirname "$GPU_LOCK_FILE")"; touch "$GPU_LOCK_FILE"
  # The CPU registration overlay is read-only and private to this container.
  docker run -d --name "$NAME" --label qwen38.submit1009.delivery=20261010 \
   --network host --ipc host --device /dev/kfd --device /dev/mkfd --device /dev/dri \
   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
   -e HIP_VISIBLE_DEVICES=6,7 -e PORT="$PORT" -e NCCL_PORT="$NCCL_PORT" \
   -e MODEL_PATH="$MODEL_PATH" -e PATCH_DIR="$PATCH_DIR" -e VARIANT="$VARIANT" -e RUN_ID="$RUN_ID" \
   -e MAX_WAIT=600 -e MAX_GATE_WAIT=5 \
   -v "$PACKAGE_DIR:$PACKAGE_DIR" -v "$MODEL_PATH:$MODEL_PATH:ro" -v /opt/hyhal:/opt/hyhal:ro \
   -v "$GPU_LOCK_FILE:/run/qwen38-gpu67.lock" \
   -v "$PACKAGE_DIR/patch/runtime/direct_register_custom_op.py:/usr/local/lib/python3.10/dist-packages/lightop/_lmslim_native/vllm_compat/direct_register_custom_op.py:ro" \
   "$IMAGE" sleep infinity
  if ! docker exec "$NAME" bash "$PACKAGE_DIR/scripts/start_server.sh"; then
   docker stop --time 20 "$NAME" >/dev/null; docker rm "$NAME" >/dev/null; exit 1
  fi
  echo "Service http://192.2.111.66:$PORT ; stop with bash scripts/run.sh stop"
  ;;
 benchmark)
  verify_container
  docker exec -e RUN_ID="$RUN_ID" "$NAME" bash "$PACKAGE_DIR/scripts/benchmark.sh" 4k
  ;;
 stop)
  if docker inspect "$NAME" >/dev/null 2>&1; then verify_container; docker stop --time 20 "$NAME"; docker rm "$NAME"; fi
  ;;
 *) echo 'Usage: run.sh {check|start|benchmark|stop}; VARIANT=candidate|baseline-v1' >&2; exit 2;;
esac
