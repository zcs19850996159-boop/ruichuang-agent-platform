#!/usr/bin/env bash
set -euo pipefail

INFRA_ROOT="/root/autodl-tmp/customer_agent_phase3/infra"
KES_BIN="$INFRA_ROOT/bin/kes"
KES_ROOT="$INFRA_ROOT/kes"
KES_CONFIG="$KES_ROOT/config.yml"
KES_CERT="$KES_ROOT/certs/server.crt"
KES_KEY="$KES_ROOT/certs/server.key"
KES_IDENTITY_FILE="$KES_ROOT/minio-identity.txt"
KES_ENV="$INFRA_ROOT/secrets/kes.env"
PHASE3C_ENV="$INFRA_ROOT/secrets/phase3c.env"
EXPECTED_SHA256="9f07258d121a69594125c6d2b569145c9b75ce80d20eaf27ab76863b689558ef"

umask 077

if [[ ! -x "$KES_BIN" ]]; then
  echo "KES binary is missing or not executable" >&2
  exit 1
fi

if [[ "$(sha256sum "$KES_BIN" | awk '{print $1}')" != "$EXPECTED_SHA256" ]]; then
  echo "KES binary checksum mismatch" >&2
  exit 1
fi

install -d -m 700 "$KES_ROOT" "$KES_ROOT/certs" "$KES_ROOT/keys" "$INFRA_ROOT/secrets"

if [[ ! -s "$KES_CERT" || ! -s "$KES_KEY" ]]; then
  "$KES_BIN" identity new \
    --ip 127.0.0.1 \
    --dns localhost \
    --expiry 87600h \
    --key "$KES_KEY" \
    --cert "$KES_CERT" \
    ruichuang-kes.local >/dev/null
fi
chmod 600 "$KES_CERT" "$KES_KEY"

if [[ ! -s "$KES_IDENTITY_FILE" ]]; then
  "$KES_BIN" identity new >"$KES_IDENTITY_FILE"
fi
chmod 600 "$KES_IDENTITY_FILE"

API_KEY="$(grep -Eo 'kes:v1:[A-Za-z0-9+/=]+' "$KES_IDENTITY_FILE" | head -n 1)"
IDENTITY="$(grep -Eio '[0-9a-f]{64}' "$KES_IDENTITY_FILE" | tail -n 1 | tr '[:upper:]' '[:lower:]')"

if [[ -z "$API_KEY" ]]; then
  echo "Unable to read KES API key" >&2
  exit 1
fi
if [[ -z "$IDENTITY" ]]; then
  IDENTITY="$("$KES_BIN" identity "$API_KEY" | grep -Eio '[0-9a-f]{64}' | tail -n 1 | tr '[:upper:]' '[:lower:]')"
fi
if [[ ! "$IDENTITY" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Unable to determine KES client identity" >&2
  exit 1
fi

cat >"$KES_CONFIG" <<EOF
address: 127.0.0.1:7373

admin:
  identity: disabled

tls:
  key: $KES_KEY
  cert: $KES_CERT

policy:
  minio-sse:
    allow:
      - /v1/api
      - /v1/status
      - /v1/metrics
      - /v1/key/create/*
      - /v1/key/generate/*
      - /v1/key/decrypt/*
    identities:
      - $IDENTITY

keystore:
  fs:
    path: $KES_ROOT/keys
EOF
chmod 600 "$KES_CONFIG"

cat >"$KES_ENV" <<EOF
MINIO_KMS_KES_ENDPOINT=https://127.0.0.1:7373
MINIO_KMS_KES_API_KEY='$API_KEY'
MINIO_KMS_KES_CAPATH=$KES_CERT
MINIO_KMS_KES_KEY_NAME=ruichuang-phase3c-default
EOF
chmod 600 "$KES_ENV"

if [[ -r "$PHASE3C_ENV" ]]; then
  ENV_TMP="$(mktemp "$INFRA_ROOT/secrets/phase3c.env.XXXXXX")"
  awk '!/^KNOWLEDGE_OBJECT_STORE_SSE=/' "$PHASE3C_ENV" >"$ENV_TMP"
  printf '%s\n' 'KNOWLEDGE_OBJECT_STORE_SSE=AES256' >>"$ENV_TMP"
  chmod 600 "$ENV_TMP"
  mv "$ENV_TMP" "$PHASE3C_ENV"
fi

echo "KES configuration prepared; identity=$IDENTITY"
