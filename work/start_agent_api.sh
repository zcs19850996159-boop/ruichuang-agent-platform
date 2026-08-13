#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REQUESTED_HOST="${HOST:-}"
REQUESTED_PORT="${PORT:-}"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env"
  set +a
fi

if [ -n "$REQUESTED_HOST" ]; then
  HOST="$REQUESTED_HOST"
fi
if [ -n "$REQUESTED_PORT" ]; then
  PORT="$REQUESTED_PORT"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p outputs/rag_agent
mkdir -p outputs/api

if [ -f "outputs/rag_agent/agent_api_${PORT}.pid" ]; then
  OLD_PID="$(cat "outputs/rag_agent/agent_api_${PORT}.pid" || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" || true
    sleep 2
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export APPLICATION_VERSION="${RELEASE_APPLICATION_VERSION:-3.6.0}"
export DEEPSEEK_TEMPERATURE="${DEEPSEEK_TEMPERATURE:-0.1}"
export API_USE_LLM_SELECTOR="${API_USE_LLM_SELECTOR:-1}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export HYBRID_INDEX_DIR="${HYBRID_INDEX_DIR:-$ROOT_DIR/outputs/rag_assets/hybrid_index_v3}"
export RETRIEVAL_CACHE_VERSION="${RETRIEVAL_CACHE_VERSION:-hybrid-v3-9cda485d4fc6-answer-v5}"

if command -v redis-server >/dev/null 2>&1 && ! redis-cli -h 127.0.0.1 -p 6379 ping >/dev/null 2>&1; then
  redis-server --bind 127.0.0.1 --port 6379 --daemonize yes \
    --dir "$ROOT_DIR/outputs/rag_agent" --dbfilename redis-state.rdb \
    --appendonly yes --appendfilename redis-state.aof --appendfsync everysec \
    --save "900 1 300 10 60 10000"
fi

nohup "$PYTHON_BIN" -m uvicorn fastapi_server:app --app-dir work --host "$HOST" --port "$PORT" \
  --timeout-keep-alive "${UVICORN_KEEP_ALIVE:-15}" --no-access-log \
  > "outputs/rag_agent/agent_api_${PORT}.out.log" \
  2> "outputs/rag_agent/agent_api_${PORT}.err.log" &

echo "$!" > "outputs/rag_agent/agent_api_${PORT}.pid"
echo "Customer agent API started on ${HOST}:${PORT}, pid=$(cat "outputs/rag_agent/agent_api_${PORT}.pid")"
