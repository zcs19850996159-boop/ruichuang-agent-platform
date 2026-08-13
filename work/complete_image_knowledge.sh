#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MANIFEST="outputs/rag_assets/manual_image_manifest.jsonl"
MANIFEST_1="outputs/rag_assets/manual_image_manifest_full_shard1.jsonl"
MANIFEST_2="outputs/rag_assets/manual_image_manifest_full_shard2.jsonl"
OUTPUT_1="outputs/rag_assets/image_knowledge_full_shard1.jsonl"
OUTPUT_2="outputs/rag_assets/image_knowledge_full_shard2.jsonl"
FINAL_OUTPUT="outputs/rag_assets/image_knowledge_auto.jsonl"
SUMMARY="outputs/rag_assets/image_knowledge_auto_summary.json"
LOG_DIR="outputs/image_knowledge"

wait_for_pid_file() {
  local pid_file="$1"
  if [ ! -f "$pid_file" ]; then
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

wait_for_pid_file "$LOG_DIR/shard1.pid"
wait_for_pid_file "$LOG_DIR/shard2.pid"

# Retry non-successful rows twice. Successful rows are skipped by source hash.
for attempt in 1 2; do
  "$PYTHON_BIN" work/build_image_knowledge.py \
    --manifest "$MANIFEST_1" --output "$OUTPUT_1" \
    --base-url http://127.0.0.1:8002/v1 \
    > "$LOG_DIR/shard1_retry${attempt}.log" 2>&1 &
  retry1=$!
  "$PYTHON_BIN" work/build_image_knowledge.py \
    --manifest "$MANIFEST_2" --output "$OUTPUT_2" \
    --base-url http://127.0.0.1:8003/v1 \
    > "$LOG_DIR/shard2_retry${attempt}.log" 2>&1 &
  retry2=$!
  wait "$retry1" "$retry2"
done

"$PYTHON_BIN" work/finalize_image_knowledge.py \
  --manifest "$MANIFEST" \
  --shards "$OUTPUT_1" "$OUTPUT_2" \
  --output "$FINAL_OUTPUT" \
  --summary "$SUMMARY" \
  > "$LOG_DIR/finalize.log" 2>&1

success_count="$("$PYTHON_BIN" -c 'import json; print(json.load(open("outputs/rag_assets/image_knowledge_auto_summary.json"))["success_count"])')"
if [ "$success_count" -ge 2580 ]; then
  "$PYTHON_BIN" work/build_hybrid_index.py \
    --output outputs/rag_assets/hybrid_index_v3 \
    --previous outputs/rag_assets/hybrid_index_v2 \
    --auto-image-knowledge "$FINAL_OUTPUT" \
    --device cuda --batch-size 48 \
    > "$LOG_DIR/build_hybrid_v3.log" 2>&1
  printf '%s\n' "ready" > "$LOG_DIR/status"
else
  printf '%s\n' "needs_review" > "$LOG_DIR/status"
fi

