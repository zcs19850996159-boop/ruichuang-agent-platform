#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$REPOSITORY_ROOT/reproduction/data/ruichuang-production-snapshot-20260813.tar.gz"
EXPECTED_SHA256="4b85d7e21fa7f7d9aecc8b697a5b37f150b16b303e3006888bdd1862c4d0f2b0"
DESTINATION="${1:-$REPOSITORY_ROOT/reproduction/restored}"

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  ACTUAL_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi

if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "快照归档 SHA-256 不匹配" >&2
  exit 1
fi

mkdir -p "$DESTINATION"
tar -xzf "$ARCHIVE" -C "$DESTINATION"
SNAPSHOT_ROOT="$DESTINATION/20260813T114434Z"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$SNAPSHOT_ROOT" && sha256sum -c SHA256SUMS)
else
  (cd "$SNAPSHOT_ROOT" && shasum -a 256 -c SHA256SUMS)
fi

printf '快照已验证并解压：%s\n' "$SNAPSHOT_ROOT"
printf '下一步请按 reproduction/README.md 恢复 PostgreSQL、RAG 和对象存储。\n'
