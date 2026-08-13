#!/usr/bin/env bash
set -euo pipefail

ARC="${1:?snapshot archive required}"
TEST_ROOT="$(mktemp -d /tmp/ruichuang-snapshot-verify.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT
tar -xzf "$ARC" -C "$TEST_ROOT"
SNAP="$(find "$TEST_ROOT" -mindepth 1 -maxdepth 1 -type d | head -1)"

printf 'ARCHIVE_SIZE='
du -h "$ARC" | awk '{print $1}'
printf 'ARCHIVE_SHA256='
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARC" | awk '{print $1}'
else
  shasum -a 256 "$ARC" | awk '{print $1}'
fi
printf 'SNAPSHOT_SIZE='
du -sh "$SNAP" | awk '{print $1}'
printf 'DB_DUMP_SIZE='
du -h "$SNAP/postgres/ruichuang_phase3c.dump" | awk '{print $1}'
if command -v pg_restore >/dev/null 2>&1; then
  pg_restore -l "$SNAP/postgres/ruichuang_phase3c.dump" >"$TEST_ROOT/pg-restore-list.txt"
  printf 'DB_TOC_ENTRIES='
  wc -l <"$TEST_ROOT/pg-restore-list.txt"
fi
python3 - "$SNAP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
objects = json.loads((root / "object_store" / "manifest.json").read_text())
inventory = json.loads((root / "inventory.json").read_text())
print(f"OBJECT_ENTRIES={len(objects['entries'])}")
print(f"MODEL_FILES={len(inventory['models'])}")
PY
printf 'FORBIDDEN_NAMES='
find "$SNAP" -type f \
  \( -name '.env' -o -iname '*secret*' -o -iname '*.pem' \
  -o -iname '*.key' -o -iname 'id_rsa' -o -iname 'id_ed25519' \) \
  -print | wc -l
printf 'CHECKSUM_VERIFY='
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$SNAP" && sha256sum -c SHA256SUMS >"$TEST_ROOT/checksums.txt")
else
  (cd "$SNAP" && shasum -a 256 -c SHA256SUMS >"$TEST_ROOT/checksums.txt")
fi
echo ok
