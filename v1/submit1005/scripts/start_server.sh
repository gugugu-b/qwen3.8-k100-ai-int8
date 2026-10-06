#!/bin/bash
# ==============================================================================
# Dedicated Model Server Launch Script for Qwen3.8-27B-Channel-INT8 (TP=2)
# Deliverable Package: submit1005
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo " Starting Optimized SGLang Model Service"
echo " Package Root: ${PACKAGE_DIR}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Environment & DTK Setup
# ------------------------------------------------------------------------------
if [ -f "/opt/dtk/env.sh" ]; then
    source /opt/dtk/env.sh
fi

export CXX=/opt/dtk/bin/hipcc
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"4,5"}
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

# rocBLAS Tensile overlay (Clean Stock DTK 26.04 tuned GEMM library)
if [ ! -d "${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean" ] && [ -f "${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean.tar.gz" ]; then
    echo "[Env] Unpacking clean rocBLAS GEMM overlay from tar.gz..."
    tar -xzf "${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean.tar.gz" -C "${PACKAGE_DIR}/config/"
fi

if [ -d "${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean" ]; then
    export ROCBLAS_TENSILE_LIBPATH="${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean"
elif [ -d "/data/wangzy/library_gpu4_int8tuned_stock_clean" ]; then
    export ROCBLAS_TENSILE_LIBPATH=/data/wangzy/library_gpu4_int8tuned_stock_clean
elif [ -d "/data/wangzy/library_gpu4_int8tuned" ]; then
    export ROCBLAS_TENSILE_LIBPATH=/data/wangzy/library_gpu4_int8tuned
fi

if [ -n "${ROCBLAS_TENSILE_LIBPATH}" ]; then
    echo "[Env] Loaded rocBLAS GEMM overlay: ${ROCBLAS_TENSILE_LIBPATH}"
    if [ -f "${ROCBLAS_TENSILE_LIBPATH}/GemmGeneralityAdd.dat" ]; then
        echo "      GemmGeneralityAdd.dat SHA256: $(sha256sum "${ROCBLAS_TENSILE_LIBPATH}/GemmGeneralityAdd.dat" | awk '{print $1}')"
    fi
fi

# Set PYTHONPATH to include our patch directory for sitecustomize.py
export PYTHONPATH="${PACKAGE_DIR}/patch:${PYTHONPATH}"
echo "[Env] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "[Env] PYTHONPATH=${PYTHONPATH}"

# ------------------------------------------------------------------------------
# 2. Check Prerequisites
# ------------------------------------------------------------------------------
if [ ! -f "/root/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch210-rocm.so" ]; then
    echo "[Notice] tvm-ffi addon not detected in /root/.cache/tvm-ffi, running apply_patch.sh first..."
    bash "${SCRIPT_DIR}/apply_patch.sh"
fi

# ------------------------------------------------------------------------------
# 3. Launch SGLang Server
# ------------------------------------------------------------------------------
PORT=${PORT:-31004}
NCCL_PORT=${NCCL_PORT:-31005}
MODEL_PATH=${MODEL_PATH:-"/data/model/Qwen3.8-27B-Channel-INT8-w8a8"}
LOG_DIR="${PACKAGE_DIR}/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/server_runtime.log"

echo "[Server] Checking port ${PORT}..."
pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
sleep 2

echo "[Server] Launching SGLang Server (TP=2) on port ${PORT}..."
echo "         Log destination: ${SERVER_LOG}"

python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size 2 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --page-size 64 \
  --served-model-name Qwen3.8-27B-Channel-INT8-w8a8 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --nccl-port "${NCCL_PORT}" \
  --pp-size 1 \
  --context-length 67584 \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --max-prefill-tokens 65536 \
  --chunked-prefill-size 8192 \
  --mamba-scheduler-strategy no_buffer \
  --cuda-graph-max-bs 32 \
  --disable-custom-all-reduce \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-metrics \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "[Server] Background process started (PID: ${SERVER_PID})."
echo "[Server] Waiting for health endpoint (http://127.0.0.1:${PORT}/health)..."

# ------------------------------------------------------------------------------
# 4. Health Check Polling
# ------------------------------------------------------------------------------
MAX_WAIT=${MAX_WAIT:-1800}
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "======================================================================"
        echo " [SUCCESS] SGLang Server is HEALTHY and READY on port ${PORT}!"
        echo " Total warmup time: ${WAIT_COUNT}s"
        echo " Next step: Run benchmarks using 'bash scripts/benchmark.sh all'"
        echo "======================================================================"
        exit 0
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
        echo "  [Warmup] Waiting for model loading & CUDA Graph capture... (${WAIT_COUNT}s / ${MAX_WAIT}s)"
    fi
done

echo "======================================================================"
echo " [ERROR] Server failed to become healthy within ${MAX_WAIT}s."
echo " Inspect logs at: ${SERVER_LOG}"
echo "======================================================================"
exit 1
