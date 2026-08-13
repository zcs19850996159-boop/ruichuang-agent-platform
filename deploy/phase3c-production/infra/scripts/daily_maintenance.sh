#!/usr/bin/env bash
set -Eeuo pipefail

PHASE3_ROOT="/root/autodl-tmp/customer_agent_phase3"
APP_ROOT="$PHASE3_ROOT/app"
SECRETS="$PHASE3_ROOT/infra/secrets/phase3c.env"
KES_SECRETS="$PHASE3_ROOT/infra/secrets/kes.env"
KES_ROOT="$PHASE3_ROOT/infra/kes"
BACKUP_ROOT="$PHASE3_ROOT/backups/daily"
PHASE2_APP="/root/autodl-tmp/customer_agent_phase2/app"
LOCK_FILE="$PHASE3_ROOT/infra/run/daily-maintenance.lock"
LOGROTATE_BIN="/usr/sbin/logrotate"

mkdir -p "$BACKUP_ROOT" "$(dirname "$LOCK_FILE")"

if [[ ! -x "$LOGROTATE_BIN" ]]; then
  echo "logrotate is required for Phase3C maintenance" >&2
  exit 1
fi

backup_once() {
  local stamp target temporary
  stamp="$(date '+%Y%m%d-%H%M%S')"
  target="$BACKUP_ROOT/$stamp"
  temporary="$BACKUP_ROOT/.$stamp.tmp"
  mkdir -p "$temporary"
  chmod 700 "$temporary"

  set -a
  source "$SECRETS"
  set +a

  /usr/lib/postgresql/14/bin/pg_dump "$CONTROL_PLANE_DATABASE_URL" \
    -Fc -f "$temporary/control-plane.dump"
  /usr/lib/postgresql/14/bin/pg_restore -l \
    "$temporary/control-plane.dump" >/dev/null

  tar -C "$PHASE3_ROOT/infra" -czf "$temporary/minio-data.tar.gz" minio-data
  tar -tzf "$temporary/minio-data.tar.gz" >/dev/null

  git -C "$APP_ROOT" bundle create "$temporary/app.git.bundle" --all
  git -C "$APP_ROOT" bundle verify "$temporary/app.git.bundle" >/dev/null

  git -C "$PHASE2_APP" bundle create "$temporary/phase2-rollback.git.bundle" --all
  git -C "$PHASE2_APP" bundle verify "$temporary/phase2-rollback.git.bundle" >/dev/null

  cp -a /root/start_ruichuang.sh "$temporary/start_ruichuang.sh"
  cp -a /etc/nginx/sites-available/ruichuang-public-6006.conf \
    "$temporary/ruichuang-public-6006.conf"
  cp -a "$SECRETS" "$temporary/phase3c.env"
  chmod 600 "$temporary/phase3c.env"

  if [[ -r "$KES_SECRETS" && -d "$KES_ROOT" ]]; then
    tar -C "$PHASE3_ROOT/infra" -czf "$temporary/kes-recovery.tar.gz" \
      kes secrets/kes.env
    tar -tzf "$temporary/kes-recovery.tar.gz" >/dev/null
    chmod 600 "$temporary/kes-recovery.tar.gz"
  fi

  (
    cd "$temporary"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum --
  ) >"$temporary/SHA256SUMS"
  chmod -R go-rwx "$temporary"
  mv "$temporary" "$target"
  printf '%s\n' "$(basename "$target")" >"$BACKUP_ROOT/LATEST_SUCCESS"
  chmod 600 "$BACKUP_ROOT/LATEST_SUCCESS"
  printf '[%s] daily backup complete: %s\n' "$(date -Is)" "$target"
}

find "$BACKUP_ROOT" -maxdepth 1 -type d -name '.*.tmp' -empty -delete

while true; do
  exec 8>"$LOCK_FILE"
  if flock -n 8; then
    backup_once
    "$LOGROTATE_BIN" /etc/logrotate.d/ruichuang-phase3c
    generation_count="$(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l)"
    if (( generation_count > 30 )); then
      printf '[%s] warning: %s local backup generations; off-host retention policy is required\n' \
        "$(date -Is)" "$generation_count" >&2
    fi
    flock -u 8
  fi
  sleep 86400
done
