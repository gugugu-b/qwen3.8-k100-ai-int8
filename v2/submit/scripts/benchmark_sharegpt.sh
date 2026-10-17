#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ID=${RUN_ID:?unique run id required}
C=${1:?client concurrency}
case "$C" in 1|4|16|32|64) ;; *) exit 2;; esac
OUT="$ROOT/result/runs/$RUN_ID/sharegpt-c$C"
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT"; exit 2; }
DATASET_PATH=${DATASET_PATH:-/data/wangzy/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}
MODEL_PATH=${MODEL_PATH:-/data/model/Qwen3.8-27B-Channel-INT8-w8a8}
python3 - "$DATASET_PATH" <<'CHECK'
import hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]); assert hashlib.sha256(p.read_bytes()).hexdigest()=='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
CHECK
mkdir -p "$OUT"

export NO_PROXY=127.0.0.1,localhost
curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:${PORT:-31014}/health" >/dev/null
curl --noproxy '*' -fsS --max-time 10 "http://127.0.0.1:${PORT:-31014}/get_server_info" > "$OUT/server-info.json"
printf 'seed=20260912; dataset=%s; C=%s; prompts=1000; full_warmup=1; rounds=3; short_warmup=64\n' "$DATASET_PATH" "$C" > "$OUT/command.txt"
FULL_WARMUP_ROUNDS=1
MEASURE_ROUNDS=3
for I in $(seq 1 "$((FULL_WARMUP_ROUNDS+MEASURE_ROUNDS))"); do
 if (( I <= FULL_WARMUP_ROUNDS )); then
  printf -v ID 'full-warmup-%02d' "$I"
 else
  printf -v ID 'round-%02d' "$((I-FULL_WARMUP_ROUNDS))"
 fi
 python3 -u -m sglang.bench_serving --backend sglang-oai --base-url "http://127.0.0.1:${PORT:-31014}" \
  --dataset-name sharegpt --dataset-path "$DATASET_PATH" \
  --model "$MODEL_PATH" --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --sharegpt-context-len 67584 --num-prompts 1000 --max-concurrency "$C" \
  --request-rate inf --warmup-requests 64 --seed 20260912 --disable-tqdm --output-details \
  --output-file "$OUT/$ID.jsonl" 2>&1 | tee "$OUT/$ID.log"
done
python3 "$ROOT/scripts/validate_sharegpt.py" "$OUT" --rounds "$MEASURE_ROUNDS" --full-warmup-rounds "$FULL_WARMUP_ROUNDS"
