#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Leave the service available for recording and inspection; explicit stop after Demo.
bash "$ROOT/scripts/run.sh" start
bash "$ROOT/scripts/run.sh" demo "${1:-请用中文简要介绍你的能力。}"
echo 'After recording, stop with: bash scripts/run.sh stop'
