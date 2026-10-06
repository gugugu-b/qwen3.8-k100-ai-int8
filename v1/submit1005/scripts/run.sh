#!/bin/bash
# ==============================================================================
# All-In-One Pipeline Execution Script for Qwen3.8-27B on Hygon K100-AI
# Deliverable Package: submit1005
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo " Starting Full Optimization Pipeline (submit1005)"
echo " Package Root: ${PACKAGE_DIR}"
echo "======================================================================"

# Step 1: Apply all patches
echo -e "\n>>> [Pipeline Step 1/3] Applying optimization patches..."
bash "${SCRIPT_DIR}/apply_patch.sh"

# Step 2: Start server and wait for readiness
echo -e "\n>>> [Pipeline Step 2/3] Starting model service and verifying health..."
bash "${SCRIPT_DIR}/start_server.sh"

# Step 3: Run benchmark suite
echo -e "\n>>> [Pipeline Step 3/3] Running benchmark suite..."
bash "${SCRIPT_DIR}/benchmark.sh" "$@"

echo "======================================================================"
echo " All-In-One Pipeline Completed Successfully!"
echo "======================================================================"
