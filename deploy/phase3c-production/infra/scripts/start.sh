#!/usr/bin/env bash
set -euo pipefail

INFRA_ROOT="/root/autodl-tmp/customer_agent_phase3/infra"
PG_DATA="/var/lib/postgresql/phase3c"
PG_LOG="/var/log/postgresql/phase3c.log"
MINIO_BIN="$INFRA_ROOT/bin/minio"
MINIO_DATA="$INFRA_ROOT/minio-data"
MINIO_LOG="$INFRA_ROOT/logs/minio.log"
MINIO_PID="$INFRA_ROOT/run/minio.pid"
SECRETS="$INFRA_ROOT/secrets/phase3c.env"
KES_SECRETS="$INFRA_ROOT/secrets/kes.env"

if [[ ! -r "$SECRETS" ]]; then
  echo "Phase3C secrets file is missing" >&2
  exit 1
fi

set -a
source "$SECRETS"
if [[ -r "$KES_SECRETS" ]]; then
  source "$KES_SECRETS"
fi
set +a

kes_ready() {
  if [[ ! -r "$KES_SECRETS" ]]; then
    return 0
  fi
  SSL_CERT_FILE="$MINIO_KMS_KES_CAPATH" \
    KES_SERVER="$MINIO_KMS_KES_ENDPOINT" \
    KES_API_KEY="$MINIO_KMS_KES_API_KEY" \
    "$INFRA_ROOT/bin/kes" status --json >/dev/null 2>&1
}

if [[ -r "$KES_SECRETS" ]]; then
  for _ in $(seq 1 30); do
    kes_ready && break
    sleep 1
  done
  if ! kes_ready; then
    echo "KES is not ready; refusing to start encrypted MinIO" >&2
    exit 1
  fi
fi

if ! /usr/lib/postgresql/14/bin/pg_isready \
  -h 127.0.0.1 -p 55432 -d ruichuang_phase3c >/dev/null 2>&1; then
  runuser -u postgres -- /usr/lib/postgresql/14/bin/pg_ctl \
    -D "$PG_DATA" -l "$PG_LOG" start
fi

minio_ready() {
  curl -fsS --max-time 3 http://127.0.0.1:59000/minio/health/live >/dev/null 2>&1
}

if ! minio_ready; then
  if [[ -r "$MINIO_PID" ]] && kill -0 "$(cat "$MINIO_PID")" 2>/dev/null; then
    kill "$(cat "$MINIO_PID")" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$(cat "$MINIO_PID")" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
  fi
  nohup env \
    MINIO_ROOT_USER="$MINIO_ROOT_USER" \
    MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
    MINIO_KMS_KES_ENDPOINT="${MINIO_KMS_KES_ENDPOINT:-}" \
    MINIO_KMS_KES_API_KEY="${MINIO_KMS_KES_API_KEY:-}" \
    MINIO_KMS_KES_CAPATH="${MINIO_KMS_KES_CAPATH:-}" \
    MINIO_KMS_KES_KEY_NAME="${MINIO_KMS_KES_KEY_NAME:-}" \
    "$MINIO_BIN" server "$MINIO_DATA" \
      --address 127.0.0.1:59000 \
      --console-address 127.0.0.1:59001 \
      >"$MINIO_LOG" 2>&1 &
  echo "$!" >"$MINIO_PID"
  chmod 600 "$MINIO_PID" "$MINIO_LOG"
fi

for _ in $(seq 1 30); do
  if /usr/lib/postgresql/14/bin/pg_isready \
      -h 127.0.0.1 -p 55432 -d ruichuang_phase3c >/dev/null 2>&1 \
    && curl -fsS http://127.0.0.1:59000/minio/health/live >/dev/null; then
    echo "Phase3C PostgreSQL and MinIO are ready on loopback"
    exit 0
  fi
  sleep 1
done

echo "Phase3C infrastructure did not become ready" >&2
exit 1
