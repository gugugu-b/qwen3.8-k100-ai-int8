#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SCENE=${1:?4k or 64k}
RUN_ID=${RUN_ID:?unique run id required}
case "$SCENE" in 4k) IN=4096; WARM=2; TIMEOUT=3600;; 64k) IN=65536; WARM=1; TIMEOUT=7200;; *) exit 2;; esac
WARM=${WARMUP_ROUNDS:-$WARM}
ROUNDS=${MEASURE_ROUNDS:-3}
RESULT_ROOT=${RESULT_ROOT:-$PACKAGE_DIR/result}
OUT="$RESULT_ROOT/$RUN_ID/$SCENE"
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:${PORT:-31014}/health" >/dev/null
python3 -u "$SCRIPT_DIR/benchmark.py" --base-url "http://127.0.0.1:${PORT:-31014}" \
 --model Qwen3.8-27B-Channel-INT8-w8a8 --output-dir "$OUT" \
 --input-len "$IN" --output-len 1024 --concurrency 8 --seed 20260912 \
 --warmup-rounds "$WARM" --measure-rounds "$ROUNDS" --requests-per-round 16 --timeout "$TIMEOUT" \
 2>&1 | tee "$OUT/bench.log"
python3 "$SCRIPT_DIR/validate_results.py" "$OUT" --rounds "$ROUNDS" --input-len "$IN"
