#!/bin/zsh
set -euo pipefail
SCRIPT_DIR=${0:A:h}
COMMAND=${1:-prepare}
/usr/bin/python3 "$SCRIPT_DIR/final_demo_control.py" "$COMMAND"
echo
echo "演示准备完成。"
