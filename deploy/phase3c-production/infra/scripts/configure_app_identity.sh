#!/usr/bin/env bash
set -Eeuo pipefail

INFRA_ROOT="/root/autodl-tmp/customer_agent_phase3/infra"
SECRETS="$INFRA_ROOT/secrets/phase3c.env"
POLICY_FILE="$INFRA_ROOT/policies/phase3c-app-s3.json"
MC_BIN="${MC_BIN:-$(command -v mcli || command -v mc || true)}"
MC_CONFIG_DIR="$(mktemp -d "$INFRA_ROOT/run/mcli-config.XXXXXX")"
trap 'rm -rf "$MC_CONFIG_DIR"' EXIT

[[ -x "$MC_BIN" ]] || { echo "MinIO client is required" >&2; exit 1; }
[[ -r "$SECRETS" ]] || { echo "Phase3C secrets are required" >&2; exit 1; }
[[ -r "$POLICY_FILE" ]] || { echo "application policy is required" >&2; exit 1; }

set -a
source "$SECRETS"
set +a

[[ "$AWS_ACCESS_KEY_ID" != "$MINIO_ROOT_USER" ]] || {
  echo "application access key must differ from MinIO root" >&2
  exit 1
}

export MC_CONFIG_DIR
"$MC_BIN" alias set phase3-admin "$KNOWLEDGE_OBJECT_STORE_ENDPOINT" \
  "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" --api S3v4 --path on >/dev/null
"$MC_BIN" admin user add phase3-admin "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" >/dev/null
"$MC_BIN" admin policy create phase3-admin ruichuang-phase3c-app "$POLICY_FILE" >/dev/null
"$MC_BIN" admin policy attach phase3-admin ruichuang-phase3c-app \
  --user "$AWS_ACCESS_KEY_ID" >/dev/null
"$MC_BIN" admin user info phase3-admin "$AWS_ACCESS_KEY_ID"
