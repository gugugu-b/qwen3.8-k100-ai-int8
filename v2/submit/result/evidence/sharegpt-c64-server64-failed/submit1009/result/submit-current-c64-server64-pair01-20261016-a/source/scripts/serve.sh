#!/usr/bin/env bash
set -eo pipefail
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/dtk/env.sh
export CXX=/opt/dtk/bin/hipcc HIP_VISIBLE_DEVICES=${PHYSICAL_PAIR:?} HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False SGLANG_USE_CUDA_IPC_TRANSPORT=1 SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_QUICK_CUSTOM_ALLREDUCE=false
export ROCBLAS_TENSILE_LIBPATH="/data/wangzy/submit1009/config/library_gpu4_int8tuned_stock_clean"
PATCH_DIR=${PATCH_DIR:-$PACKAGE_DIR/patch}
export PYTHONPATH="$PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
args=(--model-path "${MODEL_PATH:-/data/model/Qwen3.8-27B-Channel-INT8-w8a8}"
 --tp-size 2 --pp-size 1 --attention-backend fa3 --mm-attention-backend fa3
 --page-size 64 --served-model-name Qwen3.8-27B-Channel-INT8-w8a8
 --host 0.0.0.0 --port "${PORT:-31014}" --nccl-port "${NCCL_PORT:-31015}"
 --context-length 67584 --disable-radix-cache --mem-fraction-static 0.85
 --max-running-requests 64 --max-prefill-tokens 65536 --chunked-prefill-size 8192
 --mamba-scheduler-strategy no_buffer --cuda-graph-max-bs 64
 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-metrics
 --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2)
if [[ ${VARIANT:-candidate} == candidate ]]; then
 export SGLANG_USE_1STAGE_ALLREDUCE=1
 args+=(--custom-all-reduce-backend native)
else
 unset SGLANG_USE_1STAGE_ALLREDUCE
 args+=(--disable-custom-all-reduce)
fi
printf 'RUN:'; printf ' %q' python3 -u -m sglang.launch_server "${args[@]}"; printf '\n'
exec python3 -u -m sglang.launch_server "${args[@]}"
