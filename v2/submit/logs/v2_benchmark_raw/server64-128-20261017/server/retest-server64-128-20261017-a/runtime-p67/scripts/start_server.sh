#!/usr/bin/env bash
# Start only a new process group owned by this invocation; never pkill a server.
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=/data/wangzy/v2/submit
RUN_ID=${RUN_ID:?unique run id required}
PORT=${PORT:-31014}
NCCL_PORT=${NCCL_PORT:-31015}
PATCH_DIR=${PATCH_DIR:-$PACKAGE_DIR/patch}
# An experiment may hold startup while completing numerical gates. This wait
# precedes model startup and every measured/warmup request.
for ((GATE_WAITED=0; GATE_WAITED<${MAX_GATE_WAIT:-7200}; GATE_WAITED+=5)); do
 if ! compgen -G "$PATCH_DIR/start*.pending" >/dev/null; then break; fi
 if (( GATE_WAITED % 60 == 0 )); then echo "Waiting for candidate preflight gates in $PATCH_DIR"; fi
 sleep 5
done
if compgen -G "$PATCH_DIR/start*.pending" >/dev/null; then echo 'Candidate gate timed out' >&2; exit 2; fi
python3 - "$PORT" "$NCCL_PORT" <<'PY'
import socket,sys
for port in map(int,sys.argv[1:]):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(('0.0.0.0',port))
        except OSError as e: raise SystemExit(f'Port {port} is occupied: {e}')
PY
LOG_DIR="$PACKAGE_DIR/logs/$RUN_ID"
test ! -e "$LOG_DIR" || { echo "Refusing to reuse server run id: $RUN_ID" >&2; exit 2; }
mkdir -p "$LOG_DIR"
setsid bash "$SCRIPT_DIR/serve.sh" > "$LOG_DIR/server.log" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/server.pid"
for ((WAITED=0; WAITED<${MAX_WAIT:-1800}; WAITED+=5)); do
 if ! kill -0 "$PID" 2>/dev/null; then tail -30 "$LOG_DIR/server.log"; exit 1; fi
 if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:$PORT/get_server_info" > "$LOG_DIR/server-info.json"
  if [[ ${VARIANT:-candidate} == candidate ]]; then
   grep -Fq '[Native BF16 AR] Installed' "$LOG_DIR/server.log"
  fi
  echo "Server ready: pid=$PID port=$PORT logs=$LOG_DIR"; exit 0
 fi
 sleep 5
done
echo "Readiness timeout: $LOG_DIR/server.log" >&2
kill -TERM -- "-$PID" 2>/dev/null || true
exit 1
