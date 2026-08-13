#!/bin/zsh
set -euo pipefail

LOCAL_PORT=${RUICHUANG_LOCAL_PORT:-18877}
SSH_PORT=${RUICHUANG_SSH_PORT:?请设置 RUICHUANG_SSH_PORT}
SSH_HOST=${RUICHUANG_SSH_HOST:?请设置 RUICHUANG_SSH_HOST，例如 root@example.com}
PHASE3_KEY=${RUICHUANG_PHASE3_KEY:?请设置 RUICHUANG_PHASE3_KEY}
PHASE2_KEY=${RUICHUANG_PHASE2_KEY:?请设置 RUICHUANG_PHASE2_KEY}

listener_pid() {
  /usr/sbin/lsof -nP -iTCP:${LOCAL_PORT} -sTCP:LISTEN -t 2>/dev/null | head -1
}

stop_managed_tunnel() {
  local pid
  pid=$(listener_pid || true)
  [[ -z "$pid" ]] && return 0
  local command
  command=$(/bin/ps -p "$pid" -o command=)
  if [[ "$command" != *"ssh "* ]] \
    || [[ "$command" != *"127.0.0.1:${LOCAL_PORT}:127.0.0.1:887"* ]] \
    || { [[ "$command" != *"$PHASE3_KEY"* ]] && [[ "$command" != *"$PHASE2_KEY"* ]]; }; then
    echo "拒绝停止未知的 ${LOCAL_PORT} 监听进程：$pid" >&2
    exit 1
  fi
  /bin/kill "$pid"
  for _ in 1 2 3 4 5; do
    [[ -z "$(listener_pid || true)" ]] && return 0
    /bin/sleep 0.2
  done
  echo "旧隧道未能按时退出" >&2
  exit 1
}

start_tunnel() {
  local key=$1
  local target_port=$2
  /usr/bin/ssh \
    -i "$key" \
    -p "$SSH_PORT" \
    -f -N \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${target_port}" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    "$SSH_HOST"
}

verify_phase3() {
  local http_status
  http_status=$(/usr/bin/curl -sS -o /tmp/ruichuang-phase3-ready.json -w '%{http_code}' \
    "http://127.0.0.1:${LOCAL_PORT}/ready")
  [[ "$http_status" = 200 ]] || {
    echo "Phase 3 readiness 验证失败：HTTP $http_status" >&2
    exit 1
  }
  echo "已恢复 Phase 3："
  echo "  工作台 http://127.0.0.1:${LOCAL_PORT}/workbench"
  echo "  控制台 http://127.0.0.1:${LOCAL_PORT}/admin"
}

verify_phase2() {
  local http_status
  http_status=$(/usr/bin/curl -sS -o /tmp/ruichuang-phase2-health.json -w '%{http_code}' \
    "http://127.0.0.1:${LOCAL_PORT}/health")
  [[ "$http_status" = 200 ]] || {
    echo "Phase 2 回滚验证失败：HTTP $http_status" >&2
    exit 1
  }
  echo "已切换到 Phase 2 回滚服务："
  echo "  旧版演示页 http://127.0.0.1:${LOCAL_PORT}/ui"
  echo "  官方兼容 API http://127.0.0.1:${LOCAL_PORT}/chat"
}

case "${1:-status}" in
  phase2)
    stop_managed_tunnel
    start_tunnel "$PHASE2_KEY" 8877
    verify_phase2
    ;;
  phase3)
    stop_managed_tunnel
    start_tunnel "$PHASE3_KEY" 8878
    verify_phase3
    ;;
  status)
    if /usr/bin/curl -fsS "http://127.0.0.1:${LOCAL_PORT}/ready" >/dev/null 2>&1; then
      echo "当前：Phase 3"
    elif /usr/bin/curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
      echo "当前：Phase 2 回滚服务"
    else
      echo "当前：演示隧道不可用"
      exit 1
    fi
    ;;
  *)
    echo "用法：$0 phase2|phase3|status" >&2
    exit 2
    ;;
esac
