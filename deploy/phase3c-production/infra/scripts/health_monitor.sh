#!/usr/bin/env bash
set -u

PHASE3_ROOT="/root/autodl-tmp/customer_agent_phase3"
STATE_FILE="$PHASE3_ROOT/infra/run/health-monitor.json"
MONITOR_SECRETS="$PHASE3_ROOT/infra/secrets/monitoring.env"
INTERVAL_SECONDS="${HEALTH_MONITOR_INTERVAL_SECONDS:-60}"
FAILURE_THRESHOLD="${HEALTH_MONITOR_FAILURE_THRESHOLD:-3}"

if [[ -r "$MONITOR_SECRETS" ]]; then
  # shellcheck disable=SC1090
  source "$MONITOR_SECRETS"
fi

failures=0
alerted=0

healthy() {
  curl -fsS --max-time 5 http://127.0.0.1:8878/ready >/dev/null 2>&1 \
    && curl -fsS --max-time 5 http://127.0.0.1:8877/health >/dev/null 2>&1 \
    && curl -fsS --max-time 5 http://127.0.0.1:6006/ready >/dev/null 2>&1 \
    && curl -fsS --max-time 5 http://127.0.0.1:59000/minio/health/live >/dev/null 2>&1 \
    && curl -fsS --max-time 5 http://127.0.0.1:8001/health | grep -q '"loaded":true' \
    && redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -qx PONG \
    && /usr/lib/postgresql/14/bin/pg_isready \
      -h 127.0.0.1 -p 55432 -d ruichuang_phase3c >/dev/null 2>&1
}

write_state() {
  local status="$1"
  local now
  now="$(date -Is)"
  printf '{"timestamp":"%s","status":"%s","consecutive_failures":%s}\n' \
    "$now" "$status" "$failures" >"$STATE_FILE.tmp"
  chmod 600 "$STATE_FILE.tmp"
  mv "$STATE_FILE.tmp" "$STATE_FILE"
  printf '[%s] status=%s consecutive_failures=%s\n' "$now" "$status" "$failures"
}

send_alert() {
  local status="$1"
  local webhook="${ALERT_WEBHOOK_URL:-}"
  [[ -n "$webhook" ]] || return 0
  curl -fsS --max-time 10 \
    -H 'Content-Type: application/json' \
    --data "{\"service\":\"ruichuang-phase3c\",\"status\":\"$status\",\"consecutive_failures\":$failures}" \
    "$webhook" >/dev/null
}

while true; do
  if healthy; then
    failures=0
    if (( alerted )); then
      send_alert recovered || true
      alerted=0
    fi
    write_state ready
  else
    failures=$((failures + 1))
    write_state degraded
    if (( failures >= FAILURE_THRESHOLD && alerted == 0 )); then
      send_alert degraded || true
      alerted=1
    fi
  fi
  sleep "$INTERVAL_SECONDS"
done
