#!/bin/zsh
set -euo pipefail
SCRIPT_DIR=${0:A:h}
"$SCRIPT_DIR/switch_demo_runtime.sh" phase3
echo
echo "Phase 3 已恢复。"
