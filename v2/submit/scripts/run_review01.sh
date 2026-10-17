#!/usr/bin/env bash
# Explicit Docker creation is documented in README; adopt only that owned container.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-check}
NAME=qwen38-v2-review01
IMAGE_ID=sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad
PORT=${PORT:-31014}
RUN_ID=${RUN_ID:-review01-$(date -u +%Y%m%dT%H%M%SZ)-$$}
case "$RUN_ID" in *[!a-zA-Z0-9_.-]*|'') echo 'Invalid RUN_ID' >&2; exit 2;; esac
verify_container() {
 docker inspect "$NAME" | python3 -I -c '
import json,sys
x=json.load(sys.stdin)[0]; e=x["Config"]["Env"]
assert (x["Config"].get("Labels") or {}).get("qwen38.v2.review01")=="20261017"
assert x["Image"]=="sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad"
assert "HIP_VISIBLE_DEVICES=0,1" in e and "GPU_PAIR=0,1" in e
assert x["HostConfig"]["NetworkMode"]=="host"
'
}
case "$ACTION" in
 check) bash "$ROOT/scripts/run.sh" check;;
 check-runtime) bash "$ROOT/scripts/run.sh" check-runtime;;
 start)
  verify_container
  docker exec -e RUN_ID="$RUN_ID" "$NAME" bash "$ROOT/scripts/start_server.sh"
  echo "Service: http://127.0.0.1:$PORT ; stop: bash scripts/run_review01.sh stop"
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
  verify_container
  docker exec "$NAME" python3 "$ROOT/scripts/demo_client.py" "${2:-请用中文简要介绍你的能力。}"
  ;;
 stop)
  if docker inspect "$NAME" >/dev/null 2>&1; then
   verify_container; docker stop --time 20 "$NAME"; docker rm "$NAME"
  fi
  ;;
 *) echo 'Usage: run_review01.sh {check|check-runtime|start|benchmark [4k|64k] [C]|sharegpt [C]|demo [prompt]|stop}' >&2; exit 2;;
esac
