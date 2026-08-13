#!/usr/bin/env bash
set -euo pipefail

DUMP="${1:?database dump required}"
PG_BIN=/usr/lib/postgresql/14/bin
TEST_ROOT=/tmp/ruichuang-reproduction-pg-test
PG_DATA="$TEST_ROOT/data"
PG_SOCKET="$TEST_ROOT/socket"
PG_LOG="$TEST_ROOT/postgres.log"
PG_PORT=55439
TEST_DUMP="$TEST_ROOT/ruichuang_phase3c.dump"

cleanup() {
  if [[ -d "$PG_DATA" ]]; then
    runuser -u postgres -- "$PG_BIN/pg_ctl" -D "$PG_DATA" stop -m fast \
      >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

cleanup
install -d -m 0700 -o postgres -g postgres "$PG_DATA" "$PG_SOCKET"
touch "$PG_LOG"
chown postgres:postgres "$PG_LOG"
cp "$DUMP" "$TEST_DUMP"
chown postgres:postgres "$TEST_DUMP"
chmod 0400 "$TEST_DUMP"
runuser -u postgres -- "$PG_BIN/initdb" -D "$PG_DATA" \
  --auth-local=trust --auth-host=trust >/dev/null
runuser -u postgres -- "$PG_BIN/pg_ctl" -D "$PG_DATA" \
  -l "$PG_LOG" -o "-p $PG_PORT -k $PG_SOCKET -h 127.0.0.1" start >/dev/null
runuser -u postgres -- "$PG_BIN/createdb" \
  -h "$PG_SOCKET" -p "$PG_PORT" ruichuang_restore_test
runuser -u postgres -- "$PG_BIN/pg_restore" \
  -h "$PG_SOCKET" -p "$PG_PORT" -d ruichuang_restore_test \
  --no-owner --no-privileges --exit-on-error "$TEST_DUMP"

TABLES=$(runuser -u postgres -- "$PG_BIN/psql" \
  -h "$PG_SOCKET" -p "$PG_PORT" -d ruichuang_restore_test -Atqc \
  "select count(*) from pg_tables where schemaname not in ('pg_catalog','information_schema')")
ROW_QUERIES=$(runuser -u postgres -- "$PG_BIN/psql" \
  -h "$PG_SOCKET" -p "$PG_PORT" -d ruichuang_restore_test -Atqc \
  "select format('select %L, count(*) from %I.%I;', schemaname || '.' || tablename, schemaname, tablename) from pg_tables where schemaname not in ('pg_catalog','information_schema') order by 1")
printf 'RESTORE_OK=1\nTABLES=%s\n' "$TABLES"
printf '%s\n' "$ROW_QUERIES" | runuser -u postgres -- "$PG_BIN/psql" \
  -h "$PG_SOCKET" -p "$PG_PORT" -d ruichuang_restore_test -Atq
