#!/bin/bash
# ==============================================================================
# Dedicated Benchmark & Quality Validation Script for Qwen3.8-27B-Channel-INT8
# Deliverable Package: submit1005
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PORT=${PORT:-31004}
BENCH_SCRIPT=${BENCH_SCRIPT:-"/data/wangzy/bench/benchmark.py"}
RESULT_DIR="${PACKAGE_DIR}/result"
LOG_DIR="${PACKAGE_DIR}/logs"
mkdir -p "${RESULT_DIR}/4k1k" "${RESULT_DIR}/64k1k" "${LOG_DIR}"

MODE="all"
MEASURE_ROUNDS_64K=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        all|4k|64k)
            MODE="$1"
            shift
            ;;
        --quick)
            MEASURE_ROUNDS_64K=1
            shift
            ;;
        --rounds-64k)
            MEASURE_ROUNDS_64K="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/benchmark.sh [all|4k|64k] [--quick] [--rounds-64k N]"
            echo "  all        Run both 4k/1k and 64k/1k benchmarks (default)"
            echo "  4k         Run only 4k/1k benchmark"
            echo "  64k        Run only 64k/1k benchmark"
            echo "  --quick    Run only 1 measured round for 64k/1k (default is 3 rounds)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1. Use --help for usage."
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo " Running Qwen3.8-27B Benchmark Suite (Mode: ${MODE})"
echo " Target Endpoint: http://127.0.0.1:${PORT}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Health Verification
# ------------------------------------------------------------------------------
echo "[Pre-flight] Verifying server health on port ${PORT}..."
if ! curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "======================================================================"
    echo " [ERROR] SGLang server is NOT healthy on port ${PORT}."
    echo " Please start the server first by running:"
    echo "   bash scripts/start_server.sh"
    echo "======================================================================"
    exit 1
fi
echo "[Pre-flight] Server is online and healthy."

# ------------------------------------------------------------------------------
# 2. Benchmark Execution
# ------------------------------------------------------------------------------
if [ "${MODE}" == "all" ] || [ "${MODE}" == "4k" ]; then
    echo "======================================================================"
    echo " [Benchmark 1] Executing 4k/1k C=8 Benchmark (2 Warmup + 3 Measured)..."
    echo "======================================================================"
    python3 -u "${BENCH_SCRIPT}" \
      --base-url "http://0.0.0.0:${PORT}" \
      --model Qwen3.8-27B-Channel-INT8-w8a8 \
      --output-dir "${RESULT_DIR}/4k1k" \
      --input-len 4096 --output-len 1024 --concurrency 8 \
      --warmup-rounds 2 --measure-rounds 3 --requests-per-round 16 --timeout 3600 \
      2>&1 | tee "${LOG_DIR}/bench_4k1k.log"
fi

if [ "${MODE}" == "all" ] || [ "${MODE}" == "64k" ]; then
    echo "======================================================================"
    echo " [Benchmark 2] Executing 64k/1k C=8 Benchmark (1 Warmup + ${MEASURE_ROUNDS_64K} Measured)..."
    echo "======================================================================"
    python3 -u "${BENCH_SCRIPT}" \
      --base-url "http://0.0.0.0:${PORT}" \
      --model Qwen3.8-27B-Channel-INT8-w8a8 \
      --output-dir "${RESULT_DIR}/64k1k" \
      --input-len 65536 --output-len 1024 --concurrency 8 \
      --warmup-rounds 1 --measure-rounds "${MEASURE_ROUNDS_64K}" --requests-per-round 16 --timeout 7200 \
      2>&1 | tee "${LOG_DIR}/bench_64k1k.log"
fi

# ------------------------------------------------------------------------------
# 3. Quality & Text Integrity Validation
# ------------------------------------------------------------------------------
echo "======================================================================"
echo " [Validation] Quality Check & Data Integrity Report"
echo "======================================================================"

python3 - <<'EOF'
import json, glob, os

scenes_to_check = []
for s in ["4k1k", "64k1k"]:
    if glob.glob(f"submit1005/result/{s}/round-*/requests.jsonl") or glob.glob(f"result/{s}/round-*/requests.jsonl"):
        scenes_to_check.append(s)

for scene in scenes_to_check:
    files = sorted(glob.glob(f"submit1005/result/{scene}/round-*/requests.jsonl"))
    if not files:
        files = sorted(glob.glob(f"result/{scene}/round-*/requests.jsonl"))
    if not files:
        continue

    print(f"\n--- Quality Validation: {scene} ---")
    expected_in = 4096 if scene == "4k1k" else 65536
    total_reqs = 0
    total_err = 0
    total_bad_len = 0
    total_repl = 0

    for f in files:
        rows = [json.loads(l) for l in open(f)]
        err = sum(1 for r in rows if r.get('error'))
        bad = sum(1 for r in rows if r['prompt_tokens'] != expected_in or r['completion_tokens'] != 1024)
        rep = sum(r.get('replacement_chars', 0) for r in rows)
        sha = len(set(r.get('output_sha256', '') for r in rows))
        round_name = os.path.basename(os.path.dirname(f))
        print(f"  {round_name:<10}: n={len(rows):<2} err={err} wrong_len={bad} repl={rep} distinct_sha={sha}/{len(rows)}")
        total_reqs += len(rows)
        total_err += err
        total_bad_len += bad
        total_repl += rep

    status = "PASS (Zero-Defect)" if (total_err == 0 and total_bad_len == 0 and total_repl == 0) else "FAIL"
    print(f"  Summary    : Total {total_reqs} requests, Overall Status: {status}")

# Also print throughput summary if summary.json exists
for scene in scenes_to_check:
    summary_file = f"submit1005/result/{scene}/summary.json"
    if not os.path.exists(summary_file):
        summary_file = f"result/{scene}/summary.json"
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            data = json.load(f)
        rounds = data.get("rounds", [])
        if rounds:
            tps_list = [r["output_throughput_tps"] for r in rounds]
            mean_tps = sum(tps_list) / len(tps_list)
            max_tps = max(tps_list)
            print(f"\n--- Throughput Summary: {scene} ---")
            print(f"  Measured rounds: {len(rounds)}")
            print(f"  Rounds sequence: {' -> '.join(f'{v:.2f}' for v in tps_list)} tok/s")
            print(f"  Mean Output TPS: {mean_tps:.2f} tok/s  (Peak: {max_tps:.2f} tok/s)")
EOF

echo "======================================================================"
echo " Benchmark suite completed successfully!"
echo "======================================================================"
