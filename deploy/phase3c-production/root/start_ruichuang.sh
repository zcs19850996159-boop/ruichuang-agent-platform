#!/usr/bin/env bash
set -Eeuo pipefail

PHASE3_ROOT="/root/autodl-tmp/customer_agent_phase3"
PHASE3_APP="$PHASE3_ROOT/app"
PHASE3_SECRETS="$PHASE3_ROOT/infra/secrets/phase3c.env"
PHASE2_APP="/root/autodl-tmp/customer_agent_phase2/app"
PUBLIC_URL="${AutoDLService6006URL:-http://127.0.0.1:6006}"
VISION_HEALTH_URL="http://127.0.0.1:8001/health"
VISION_PID_FILE="$PHASE3_APP/outputs/rag_agent/vision_qwen_8001.pid"
START_LOCK_DIR="$PHASE3_ROOT/infra/run/start-ruichuang.lock.d"
SUPERVISOR_CONF="$PHASE3_ROOT/infra/supervisor/phase3c-supervisord.conf"
SUPERVISOR_PID_FILE="$PHASE3_ROOT/infra/run/supervisord.pid"
SUPERVISOR_SOCKET="$PHASE3_ROOT/infra/run/supervisor.sock"
SUPERVISOR_START_LOCK="$PHASE3_ROOT/infra/run/supervisor-start.lock"
SUPERVISOR_LOG="$PHASE3_ROOT/infra/logs/supervisor-bootstrap.log"
SUPERVISORD="/root/miniconda3/bin/supervisord"
SUPERVISORCTL="/root/miniconda3/bin/supervisorctl"

mkdir -p "$(dirname "$START_LOCK_DIR")"
if ! mkdir "$START_LOCK_DIR" 2>/dev/null; then
  if [[ -r "$START_LOCK_DIR/pid" ]] \
    && kill -0 "$(cat "$START_LOCK_DIR/pid")" 2>/dev/null; then
    printf '[%s] 另一个启动或恢复任务正在运行，跳过重复执行\n' "$(date '+%H:%M:%S')"
    exit 0
  fi
  rm -f "$START_LOCK_DIR/pid"
  rmdir "$START_LOCK_DIR" 2>/dev/null || true
  mkdir "$START_LOCK_DIR"
fi
printf '%s\n' "$$" >"$START_LOCK_DIR/pid"
trap 'rm -f "$START_LOCK_DIR/pid"; rmdir "$START_LOCK_DIR" 2>/dev/null || true' EXIT

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

is_phase3_supervisor() {
  local pid="$1"
  local command_line=""
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$command_line" == *"$SUPERVISORD"* ]] \
    && [[ "$command_line" == *"$SUPERVISOR_CONF"* ]]
}

supervisor_controller_ready() {
  local pid=""
  pid="$("$SUPERVISORCTL" -c "$SUPERVISOR_CONF" pid 2>/dev/null || true)"
  is_phase3_supervisor "$pid"
}

ensure_phase3_supervisor() {
  local existing_pid=""
  local i

  mkdir -p "$PHASE3_ROOT/infra/run" "$PHASE3_ROOT/infra/logs"
  exec 9>"$SUPERVISOR_START_LOCK"
  flock 9

  if supervisor_controller_ready; then
    flock -u 9
    return 0
  fi

  if [[ -r "$SUPERVISOR_PID_FILE" ]]; then
    existing_pid="$(cat "$SUPERVISOR_PID_FILE")"
  fi
  if is_phase3_supervisor "$existing_pid"; then
    for i in $(seq 1 30); do
      if supervisor_controller_ready; then
        flock -u 9
        return 0
      fi
      sleep 1
    done
    log "Phase3C Supervisor 进程存在，但控制接口未就绪"
    flock -u 9
    return 1
  fi

  # A PID may be reused by an unrelated process after a container restart.
  # Clear stale state without ever signalling the process that owns that PID.
  rm -f "$SUPERVISOR_PID_FILE" "$SUPERVISOR_SOCKET"
  log "启动 Phase3C Supervisor（KES + 守护 + 监控 + 备份）"
  nohup "$SUPERVISORD" -n -c "$SUPERVISOR_CONF" \
    >"$SUPERVISOR_LOG" 2>&1 9>&- &

  for i in $(seq 1 30); do
    if supervisor_controller_ready; then
      flock -u 9
      return 0
    fi
    sleep 1
  done
  log "Phase3C Supervisor 启动失败，日志：$SUPERVISOR_LOG"
  flock -u 9
  return 1
}

require_file() {
  if [[ ! -r "$1" ]]; then
    log "缺少必要文件：$1"
    exit 1
  fi
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempts="${3:-90}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      log "$name 已就绪：$url"
      return 0
    fi
    sleep 1
  done
  log "$name 启动超时：$url"
  return 1
}

vision_ready() {
  local payload
  payload="$(curl -fsS --max-time 3 "$VISION_HEALTH_URL" 2>/dev/null || true)"
  [[ "$payload" == *'"loaded":true'* ]]
}

wait_vision() {
  local attempts="${1:-180}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if vision_ready; then
      log "Qwen 视觉服务已就绪：$VISION_HEALTH_URL"
      return 0
    fi
    sleep 1
  done
  log "Qwen 视觉服务启动超时：$VISION_HEALTH_URL"
  return 1
}

stop_pid_file() {
  local pid_file="$1"
  local pid=""
  if [[ -r "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        return 0
      fi
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

ensure_redis() {
  local redis_dir="$PHASE3_APP/outputs/rag_agent"
  mkdir -p "$redis_dir"
  if ! redis-cli -h 127.0.0.1 -p 6379 ping >/dev/null 2>&1; then
    redis-server --bind 127.0.0.1 --port 6379 --daemonize yes \
      --dir "$redis_dir" --dbfilename redis-state.rdb \
      --appendonly yes --appendfilename redis-state.aof --appendfsync everysec \
      --save "900 1 300 10 60 10000"
  fi
  redis-cli -h 127.0.0.1 -p 6379 CONFIG SET appendonly yes >/dev/null
  redis-cli -h 127.0.0.1 -p 6379 CONFIG SET appendfsync everysec >/dev/null
  redis-cli -h 127.0.0.1 -p 6379 CONFIG SET save "900 1 300 10 60 10000" >/dev/null
  redis-cli -h 127.0.0.1 -p 6379 ping | grep -qx PONG
}

require_file "$PHASE3_ROOT/infra/scripts/start.sh"
require_file "$PHASE3_APP/work/start_agent_api.sh"
require_file "$PHASE3_APP/work/start_vision_qwen.sh"
require_file "$PHASE3_APP/.env"
require_file "$PHASE3_SECRETS"
require_file "$PHASE2_APP/work/start_agent_api.sh"
require_file "$SUPERVISOR_CONF"
require_file "$SUPERVISORD"
require_file "$SUPERVISORCTL"

ensure_phase3_supervisor

if ! (
  set -a
  source "$PHASE3_SECRETS"
  source "$PHASE3_APP/.env"
  set +a
  [[ "${DEPLOYMENT_MODE:-}" != "production" ]] \
    || [[ "${AWS_ACCESS_KEY_ID:-}" != "${MINIO_ROOT_USER:-}" ]]
); then
  log "生产环境拒绝让应用复用 MinIO root 身份"
  exit 1
fi

log "启动新版基础设施（PostgreSQL + MinIO）"
bash "$PHASE3_ROOT/infra/scripts/start.sh"

log "检查 Redis 持久化状态"
ensure_redis

if vision_ready; then
  log "Qwen 视觉服务已经健康，跳过重复启动"
elif [[ -r "$VISION_PID_FILE" ]] \
  && kill -0 "$(cat "$VISION_PID_FILE")" 2>/dev/null; then
  log "Qwen 视觉服务进程存在但尚未就绪，等待 30 秒"
  if ! wait_vision 30; then
    log "Qwen 视觉服务加载失败，终止旧进程并重新启动"
    stop_pid_file "$VISION_PID_FILE"
    (
      cd "$PHASE3_APP"
      PYTHON_BIN=/root/miniconda3/bin/python bash work/start_vision_qwen.sh
    )
  fi
else
  log "启动 Qwen 视觉服务（8001）"
  (
    cd "$PHASE3_APP"
    PYTHON_BIN=/root/miniconda3/bin/python bash work/start_vision_qwen.sh
  )
fi
wait_vision 180

if curl -fsS --max-time 3 http://127.0.0.1:8878/ready >/dev/null 2>&1 \
  && curl -fsS --max-time 3 http://127.0.0.1:8878/workbench >/dev/null 2>&1 \
  && curl -fsS --max-time 3 http://127.0.0.1:8878/admin >/dev/null 2>&1; then
  log "新版 Phase 3 服务已经健康，跳过重复启动"
else
  log "启动新版 Phase 3 服务（8878）"
  (
    cd "$PHASE3_APP"
    set -a
    source "$PHASE3_SECRETS"
    source "$PHASE3_APP/.env"
    set +a
    # The API must never inherit MinIO root credentials. Infrastructure and
    # recovery scripts may use them, while the application uses AWS_* only.
    unset MINIO_ROOT_USER MINIO_ROOT_PASSWORD
    HOST=127.0.0.1 PORT=8878 bash work/start_agent_api.sh
  )
fi

if curl -fsS --max-time 3 http://127.0.0.1:8877/health >/dev/null 2>&1; then
  log "Phase 2 回退服务已经健康，跳过重复启动"
else
  log "启动 Phase 2 回退服务（8877）"
  (
    cd "$PHASE2_APP"
    HOST=127.0.0.1 PORT=8877 bash work/start_agent_api.sh
  )
fi

log "启动或重载 Nginx 公网入口（6006）"
nginx -t
if pgrep -x nginx >/dev/null 2>&1; then
  nginx -s reload
else
  nginx
fi

wait_http "新版健康检查" "http://127.0.0.1:8878/ready" 120
wait_http "新版聊天工作台" "http://127.0.0.1:8878/workbench" 20
wait_http "企业控制台" "http://127.0.0.1:8878/admin" 20
wait_http "Phase 2 回退服务" "http://127.0.0.1:8877/health" 120
wait_http "Nginx 公网代理" "http://127.0.0.1:6006/ready" 20

READY_JSON="$(curl -fsS http://127.0.0.1:8878/ready)"
READY_JSON="$READY_JSON" /root/miniconda3/bin/python -c '
import json, os
checks = json.loads(os.environ["READY_JSON"])["data"]["checks"]
assert checks["control_plane"]["status"] == "ready"
assert checks["control_plane"]["backend"] == "postgresql"
assert checks["managed_knowledge"]["status"] == "ready"
assert checks["managed_knowledge"]["backend"] == "s3"
assert checks["vision"]["ready"] is True
assert checks["state"]["ready"] is True
'

log "睿创新版系统启动完成"
printf '\n访问地址：\n'
printf '  聊天工作台：%s/workbench\n' "$PUBLIC_URL"
printf '  企业控制台：%s/admin\n' "$PUBLIC_URL"
printf '  API 文档：  仅限服务器本机 http://127.0.0.1:8878/docs\n'
printf '  健康检查：  %s/ready\n' "$PUBLIC_URL"
