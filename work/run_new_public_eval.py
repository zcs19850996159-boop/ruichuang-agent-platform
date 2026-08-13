from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any


ROW_START_RE = re.compile(r"(?m)^(\d+),")


def decode_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace"), "gb18030-replace"


def clean_question(raw: str) -> str:
    text = str(raw or "").strip("\r\n ,")
    text = text.replace('""', '"').strip('"')
    text = re.sub(r'"\s*,\s*\??\s*"', " ", text)
    text = text.replace(r"\u2019", "’")
    text = re.sub(r'^[?,]+|[",]+$', '', text).strip()
    return re.sub(r"\s+", " ", text)


def load_questions(path: Path) -> tuple[list[dict[str, str]], str]:
    text, encoding = decode_source(path)
    matches = list(ROW_START_RE.finditer(text))
    rows: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        question = clean_question(text[match.end() : end])
        if question:
            rows.append({"id": match.group(1), "question": question})
    return rows, encoding


def post_question(
    base_url: str,
    row: dict[str, str],
    *,
    token: str,
    timeout: float,
    session_prefix: str,
) -> dict[str, Any]:
    import requests

    headers = {"Content-Type": "application/json", "X-Request-Id": f"new-public-{row['id']}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/chat",
        headers=headers,
        json={"question": row["question"], "session_id": f"{session_prefix}-{row['id']}"},
        timeout=timeout,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    try:
        payload = response.json()
    except ValueError:
        payload = {"code": response.status_code, "msg": response.text[:1000], "data": None}
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    remote_media = data.get("remote_media") if isinstance(data.get("remote_media"), dict) else {}
    input_images = data.get("input_images") if isinstance(data.get("input_images"), dict) else {}
    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    return {
        "id": row["id"],
        "question": row["question"],
        "http_status": response.status_code,
        "api_code": payload.get("code") if isinstance(payload, dict) else None,
        "api_msg": payload.get("msg") if isinstance(payload, dict) else "",
        "route_type": route.get("route_type") or "",
        "manual_id": route.get("manual_id") or "",
        "answer": data.get("answer") or "",
        "images": data.get("images") or [],
        "remote_media_detected": remote_media.get("detected", 0),
        "remote_media_usable": remote_media.get("usable", 0),
        "vision_used": bool(input_images.get("used")),
        "vision_context": input_images.get("context") or "",
        "constraint_pass": (data.get("answer_check") or {}).get("constraint_pass") if isinstance(data.get("answer_check"), dict) else None,
        "source_count": len(sources),
        "retrieval_mode": retrieval.get("mode") or "",
        "top_rerank_score": retrieval.get("top_rerank_score"),
        "top_score_gap": retrieval.get("top_score_gap"),
        "retrieval_decision": retrieval.get("decision") or "",
        "images_truncated": retrieval.get("images_truncated", 0),
        "refusal_type": data.get("refusal_type") or "",
        "server_elapsed_ms": data.get("elapsed_ms"),
        "client_elapsed_ms": latency_ms,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["id", "question"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cooked = dict(row)
            cooked["images"] = json.dumps(cooked.get("images") or [], ensure_ascii=False)
            writer.writerow(cooked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the new official public questions through the customer-agent API.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-url", default=os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--token", default=os.environ.get("KAFU_API_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--session-prefix", default="new-public")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated question IDs for targeted smoke tests")
    parser.add_argument("--output-jsonl", default="outputs/rag_agent/new_public_eval_results.jsonl")
    parser.add_argument("--output-csv", default="outputs/rag_agent/new_public_eval_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows, encoding = load_questions(Path(args.input))
    if args.ids.strip():
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        rows = [row for row in rows if row["id"] in wanted]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(json.dumps({"parsed": len(rows), "encoding": encoding, "dry_run": args.dry_run}, ensure_ascii=False))
    if args.dry_run:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    for row in rows:
        try:
            result = post_question(
                args.base_url,
                row,
                token=args.token,
                timeout=args.timeout,
                session_prefix=args.session_prefix,
            )
        except Exception as exc:
            result = {
                "id": row["id"],
                "question": row["question"],
                "http_status": 0,
                "api_code": None,
                "api_msg": f"{type(exc).__name__}: {exc}",
                "route_type": "",
                "manual_id": "",
                "answer": "",
                "images": [],
                "remote_media_detected": 0,
                "remote_media_usable": 0,
                "vision_used": False,
                "vision_context": "",
                "constraint_pass": None,
                "server_elapsed_ms": None,
                "client_elapsed_ms": None,
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    jsonl_path = Path(args.output_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    write_csv(Path(args.output_csv), results)
    summary = {
        "total": len(results),
        "api_ok": sum(row.get("http_status") == 200 and row.get("api_code") == 0 for row in results),
        "remote_media_questions": sum(bool(row.get("remote_media_detected")) for row in results),
        "remote_media_usable": sum(bool(row.get("remote_media_usable")) for row in results),
        "vision_used": sum(bool(row.get("vision_used")) for row in results),
        "constraint_pass": sum(row.get("constraint_pass") is True for row in results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
