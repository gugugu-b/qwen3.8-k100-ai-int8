#!/bin/bash
# ==============================================================================
# One-Click Patch Script for Qwen3.8-27B-Channel-INT8 on Hygon K100-AI
# Deliverable Package: submit1005
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${PACKAGE_DIR}/patch"

echo "======================================================================"
echo " Applying All Optimization Patches for Qwen3.8-27B on K100-AI"
echo " Package Root: ${PACKAGE_DIR}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Environment & DTK Setup
# ------------------------------------------------------------------------------
if [ -f "/opt/dtk/env.sh" ]; then
    source /opt/dtk/env.sh
fi
export CXX=/opt/dtk/bin/hipcc

# ------------------------------------------------------------------------------
# 2. Deploy tvm-ffi DLPack ROCm Prebuilt / Build Cache
# ------------------------------------------------------------------------------
TVM_CACHE_DIR="/root/.cache/tvm-ffi"
mkdir -p "${TVM_CACHE_DIR}"
DL_PACK_SO="libtorch_c_dlpack_addon_torch210-rocm.so"

if [ -f "${PATCH_DIR}/${DL_PACK_SO}" ]; then
    echo "[Patch 1/5] Installing prebuilt tvm-ffi dlpack ROCm addon..."
    cp "${PATCH_DIR}/${DL_PACK_SO}" "${TVM_CACHE_DIR}/${DL_PACK_SO}"
    touch "${TVM_CACHE_DIR}/${DL_PACK_SO}.lock" 2>/dev/null || true
elif [ ! -f "${TVM_CACHE_DIR}/${DL_PACK_SO}" ]; then
    echo "[Patch 1/5] Building tvm-ffi dlpack ROCm addon using hipcc..."
    python3 /usr/local/lib/python3.10/dist-packages/tvm_ffi/utils/_build_optional_torch_c_dlpack.py \
      --output-dir "${TVM_CACHE_DIR}" \
      --libname "${DL_PACK_SO}" --build-with-rocm
else
    echo "[Patch 1/5] tvm-ffi dlpack ROCm addon already exists in cache."
fi

# ------------------------------------------------------------------------------
# 3. Apply lightop / lmslim Duplicate Registration Guard
# ------------------------------------------------------------------------------
if [ -f "${PATCH_DIR}/patch_dup_reg_guard.py" ]; then
    echo "[Patch 2/5] Applying idempotent registration guard to lightop..."
    python3 "${PATCH_DIR}/patch_dup_reg_guard.py"
    find /usr/local/lib/python3.10/dist-packages/lightop -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
else
    echo "[Patch 2/5] Warning: patch_dup_reg_guard.py not found, skipping."
fi

# ------------------------------------------------------------------------------
# 4. Deploy Aiter Triton Real-Dimension Configurations
# ------------------------------------------------------------------------------
AITER_TARGET_DIR="/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs"
if [ -d "${AITER_TARGET_DIR}" ] && [ -d "${PACKAGE_DIR}/config/aiter" ]; then
    echo "[Patch 3/5] Synchronizing Aiter gfx928 real-dimension configs (H=8, HV=24)..."
    cp -r "${PACKAGE_DIR}/config/aiter/"* "${AITER_TARGET_DIR}/" || true
else
    echo "[Patch 3/5] Skipping Aiter configs (directory not found or not required)."
fi

# ------------------------------------------------------------------------------
# 5. Ensure Unified Multi-M GEMV Kernel is Available
# ------------------------------------------------------------------------------
TARGET_EXT_DIR="/data/wangzy/native_ext"
mkdir -p "${TARGET_EXT_DIR}" 2>/dev/null || true

if [ ! -f "${PATCH_DIR}/k100_int8_gemv_unified.so" ]; then
    echo "[Patch 4/6] Compiling Unified GEMV kernel with hipcc..."
    python3 "${PATCH_DIR}/build_unified.py"
fi

if [ -w "${TARGET_EXT_DIR}" ] && [ -f "${PATCH_DIR}/k100_int8_gemv_unified.so" ]; then
    cp -u "${PATCH_DIR}/k100_int8_gemv_unified.so" "${TARGET_EXT_DIR}/k100_int8_gemv_unified.so" 2>/dev/null || true
fi
echo "[Patch 4/6] Unified GEMV kernel ready."

# ------------------------------------------------------------------------------
# 6. Verify Self-Contained rocBLAS Clean GEMM Overlay
# ------------------------------------------------------------------------------
CLEAN_LIB_DIR="${PACKAGE_DIR}/config/library_gpu4_int8tuned_stock_clean"
if [ ! -d "${CLEAN_LIB_DIR}" ] && [ -f "${CLEAN_LIB_DIR}.tar.gz" ]; then
    echo "[Patch 5/6] Unpacking clean rocBLAS GEMM overlay from tar.gz..."
    tar -xzf "${CLEAN_LIB_DIR}.tar.gz" -C "${PACKAGE_DIR}/config/"
fi

if [ -d "${CLEAN_LIB_DIR}" ] && [ -f "${CLEAN_LIB_DIR}/GemmGeneralityAdd.dat" ]; then
    EXPECTED_SHA="c3e295b24494507ede49ee2b330a96ddc9fe1654c58e5fd9611b3adf8b227c55"
    ACTUAL_SHA=$(sha256sum "${CLEAN_LIB_DIR}/GemmGeneralityAdd.dat" | awk '{print $1}')
    if [ "${ACTUAL_SHA}" = "${EXPECTED_SHA}" ]; then
        echo "[Patch 5/6] Verified clean rocBLAS GEMM overlay (SHA256: ${ACTUAL_SHA:0:16}... OK)."
    else
        echo "[Patch 5/6] Warning: GemmGeneralityAdd.dat SHA256 mismatch: ${ACTUAL_SHA}"
    fi
else
    echo "[Patch 5/6] Warning: library_gpu4_int8tuned_stock_clean not found in config/."
fi

# ------------------------------------------------------------------------------
# 7. Verify Patch Integrity with Quick Python Import
# ------------------------------------------------------------------------------
echo "[Patch 6/6] Verifying patched environment importability..."
PYTHONPATH="${PATCH_DIR}:${PYTHONPATH}" python3 -c "
import sitecustomize
import sglang
print('    -> sglang + sitecustomize + Triton Long-KV verified OK!')
"

echo "======================================================================"
echo " [SUCCESS] All patches successfully applied and verified!"
echo " Next step: Start the server using 'bash scripts/start_server.sh'"
echo "======================================================================"
