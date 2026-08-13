#!/bin/zsh
set -euo pipefail
SCRIPT_DIR=${0:A:h}
"$SCRIPT_DIR/switch_demo_runtime.sh" phase2
echo
echo "Phase 2 回滚完成；请使用上方旧版演示页地址。"
