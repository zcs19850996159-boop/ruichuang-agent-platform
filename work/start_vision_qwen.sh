#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${VISION_LOCAL_HOST:-127.0.0.1}"
PORT="${VISION_LOCAL_PORT:-8001}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${VISION_LOCAL_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
MODEL_NAME="${VISION_LOCAL_MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"

mkdir -p outputs/rag_agent /root/autodl-tmp/huggingface

PID_FILE="outputs/rag_agent/vision_qwen_${PORT}.pid"
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" || true
    sleep 3
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export VISION_LOCAL_MAX_NEW_TOKENS="${VISION_LOCAL_MAX_NEW_TOKENS:-256}"
export VISION_LOCAL_MAX_REQUEST_BYTES="${VISION_LOCAL_MAX_REQUEST_BYTES:-16777216}"

nohup "$PYTHON_BIN" work/vision_qwen_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  > "outputs/rag_agent/vision_qwen_${PORT}.out.log" \
  2> "outputs/rag_agent/vision_qwen_${PORT}.err.log" &

echo "$!" > "$PID_FILE"
echo "Qwen vision service starting on ${HOST}:${PORT}, pid=$(cat "$PID_FILE")"
