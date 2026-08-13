#!/usr/bin/env bash
set -Eeuo pipefail

CONF="/root/autodl-tmp/customer_agent_phase3/infra/supervisor/phase3c-supervisord.conf"
RUN_DIR="/root/autodl-tmp/customer_agent_phase3/infra/run"
LOG_DIR="/root/autodl-tmp/customer_agent_phase3/infra/logs"
PID_FILE="$RUN_DIR/supervisord.pid"
SOCKET_FILE="$RUN_DIR/supervisor.sock"
SUPERVISORD="/root/miniconda3/bin/supervisord"
SUPERVISORCTL="/root/miniconda3/bin/supervisorctl"
START_LOCK="$RUN_DIR/supervisor-start.lock"

mkdir -p "$RUN_DIR" "$LOG_DIR"
exec 9>"$START_LOCK"
flock 9

is_phase3_supervisor() {
  local pid="$1"
  local command_line=""
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  [[ "$command_line" == *"$SUPERVISORD"* ]] \
    && [[ "$command_line" == *"$CONF"* ]]
}

controller_ready() {
  local pid=""
  pid="$("$SUPERVISORCTL" -c "$CONF" pid 2>/dev/null || true)"
  is_phase3_supervisor "$pid"
}

if controller_ready; then
  flock -u 9
  exec 9>&-
  exit 0
fi

if [[ -r "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if is_phase3_supervisor "$existing_pid"; then
    for _ in $(seq 1 30); do
      if controller_ready; then
        flock -u 9
        exec 9>&-
        exit 0
      fi
      sleep 1
    done
    echo "Phase3C Supervisor is running but its control socket is not ready" >&2
    flock -u 9
    exec 9>&-
    exit 1
  fi
fi

# PID values can be reused after a container restart. Remove only stale state;
# never signal the unrelated process that happens to own the old numeric PID.
rm -f "$PID_FILE" "$SOCKET_FILE"
flock -u 9
exec 9>&-

exec "$SUPERVISORD" -n -c "$CONF"
