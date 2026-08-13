#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
VISION_PORT="${VISION_LOCAL_PORT:-8001}"
AGENT_PORT="${PORT:-8765}"
PUBLIC_AGENT_PORT="${PUBLIC_PORT:-6006}"

bash work/start_vision_qwen.sh

for _ in $(seq 1 120); do
  if "$PYTHON_BIN" - <<PY >/tmp/customer_agent_vision_health.out 2>/tmp/customer_agent_vision_health.err
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${VISION_PORT}/health", timeout=2).read()
PY
  then
    break
  fi
  sleep 2
done

"$PYTHON_BIN" - <<PY
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:${VISION_PORT}/health", timeout=5) as resp:
    health = json.loads(resp.read().decode("utf-8"))
if not health.get("data", {}).get("loaded"):
    raise SystemExit("vision service did not become ready")
print("vision ready:", json.dumps(health.get("data", {}), ensure_ascii=False))
PY

bash work/start_agent_api.sh

if [ -n "$PUBLIC_AGENT_PORT" ] && [ "$PUBLIC_AGENT_PORT" != "$AGENT_PORT" ]; then
  PORT="$PUBLIC_AGENT_PORT" bash work/start_agent_api.sh
fi

for _ in $(seq 1 60); do
  if "$PYTHON_BIN" - <<PY >/tmp/customer_agent_api_health.out 2>/tmp/customer_agent_api_health.err
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${AGENT_PORT}/health", timeout=2).read()
PY
  then
    break
  fi
  sleep 2
done

"$PYTHON_BIN" - <<PY
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:${AGENT_PORT}/health", timeout=5) as resp:
    health = json.loads(resp.read().decode("utf-8"))
data = health.get("data", {})
if not data.get("image_input_enabled"):
    raise SystemExit("agent API is running, but image_input_enabled is false")
print("agent ready:", json.dumps(data, ensure_ascii=False))
PY

if [ -n "$PUBLIC_AGENT_PORT" ] && [ "$PUBLIC_AGENT_PORT" != "$AGENT_PORT" ]; then
  "$PYTHON_BIN" - <<PY
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:${PUBLIC_AGENT_PORT}/health", timeout=5) as resp:
    health = json.loads(resp.read().decode("utf-8"))
data = health.get("data", {})
if not data.get("image_input_enabled"):
    raise SystemExit("public-port agent API is running, but image_input_enabled is false")
print("public agent ready:", json.dumps(data, ensure_ascii=False))
PY
fi
