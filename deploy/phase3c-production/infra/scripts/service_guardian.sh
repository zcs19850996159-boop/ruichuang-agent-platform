#!/usr/bin/env bash
set -u

START_SCRIPT="/root/start_ruichuang.sh"
APP_ROOT="/root/autodl-tmp/customer_agent_phase3/app"
SECRETS="/root/autodl-tmp/customer_agent_phase3/infra/secrets/phase3c.env"
KES_SECRETS="/root/autodl-tmp/customer_agent_phase3/infra/secrets/kes.env"

kes_healthy() {
  if [[ ! -r "$KES_SECRETS" ]]; then
    return 0
  fi
  # shellcheck disable=SC1090
  source "$KES_SECRETS"
  SSL_CERT_FILE="$MINIO_KMS_KES_CAPATH" \
    KES_SERVER="$MINIO_KMS_KES_ENDPOINT" \
    KES_API_KEY="$MINIO_KMS_KES_API_KEY" \
    /root/autodl-tmp/customer_agent_phase3/infra/bin/kes \
      status --json >/dev/null 2>&1
}

healthy() {
  kes_healthy \
    && curl -fsS --max-time 4 http://127.0.0.1:8878/ready >/dev/null 2>&1 \
    && curl -fsS --max-time 4 http://127.0.0.1:8877/health >/dev/null 2>&1 \
    && curl -fsS --max-time 4 http://127.0.0.1:6006/ready >/dev/null 2>&1 \
    && curl -fsS --max-time 4 http://127.0.0.1:59000/minio/health/live >/dev/null 2>&1 \
    && curl -fsS --max-time 4 http://127.0.0.1:8001/health | grep -q '"loaded":true' \
    && redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -qx PONG \
    && /usr/lib/postgresql/14/bin/pg_isready \
      -h 127.0.0.1 -p 55432 -d ruichuang_phase3c >/dev/null 2>&1
}

recover() {
  printf '[%s] dependency check failed; running recovery\n' "$(date -Is)"
  timeout 360 bash "$START_SCRIPT"
}

if [[ ! -r "$START_SCRIPT" || ! -r "$APP_ROOT/.env" || ! -r "$SECRETS" ]]; then
  printf '[%s] guardian configuration is incomplete\n' "$(date -Is)" >&2
  exit 1
fi

if ! healthy; then
  recover || true
fi

while true; do
  sleep 15
  if ! healthy; then
    recover || true
  fi
done
