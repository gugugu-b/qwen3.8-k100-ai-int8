#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SCENE=${1:?4k or 64k}; C=${2:-8}; RUN_ID=${RUN_ID:?unique run id required}
case "$SCENE" in 4k) IN=4096; WARM=2; TIMEOUT=3600;; 64k) IN=65536; WARM=1; TIMEOUT=7200;; *) exit 2;; esac
[[ "$C" =~ ^[1-9][0-9]*$ ]] && (( C <= 128 )) || exit 2
OUT="$ROOT/result/runs/$RUN_ID/$SCENE-c$C"
test ! -e "$OUT" || { echo "Refusing overwrite: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
export NO_PROXY=127.0.0.1,localhost
BASE="http://127.0.0.1:${PORT:-31014}"
curl --noproxy '*' -fsS --max-time 10 "$BASE/health" >/dev/null
curl --noproxy '*' -fsS --max-time 10 "$BASE/get_server_info" > "$OUT/server-info.json"
REQUESTS=$((2*C)); ((REQUESTS>=16)) || REQUESTS=16
cmd=(python3 -u "$SCRIPT_DIR/benchmark.py" --base-url "$BASE" --model Qwen3.8-27B-Channel-INT8-w8a8
 --output-dir "$OUT" --input-len "$IN" --output-len 1024 --concurrency "$C" --seed 20260912
 --warmup-rounds "$WARM" --measure-rounds 3 --requests-per-round "$REQUESTS" --timeout "$TIMEOUT")
printf '%q ' "${cmd[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
"${cmd[@]}" 2>&1 | tee "$OUT/bench.log"
python3 "$SCRIPT_DIR/validate_results.py" "$OUT" --rounds 3 --input-len "$IN" --requests "$REQUESTS"
