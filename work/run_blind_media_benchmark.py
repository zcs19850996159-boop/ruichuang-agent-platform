from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs" / "blind_media_benchmark_v1" / "manifest.jsonl"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--token", default=os.environ.get("KAFU_API_TOKEN", ""))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--reviewed-rubric", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "blind_media_benchmark_v1" / "results.jsonl"),
    )
    args = parser.parse_args()

    health = requests.get(args.base_url.rstrip("/") + "/health", timeout=20).json()
    health_data = health.get("data") if isinstance(health, dict) else {}
    grounding = health_data.get("verified_visual_grounding") if isinstance(health_data, dict) else {}
    if not isinstance(grounding, dict) or grounding.get("enabled") is not False:
        raise SystemExit(
            "Blind benchmark refused: verified_visual_grounding must be disabled on the target service."
        )

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.reviewed_rubric:
        reviewed_payload = json.loads(Path(args.reviewed_rubric).read_text(encoding="utf-8"))
        reviewed_by_id = {
            str(case.get("case_id")): case for case in (reviewed_payload.get("cases") or [])
        }
        for row in rows:
            reviewed = reviewed_by_id.get(str(row.get("case_id")))
            if reviewed and reviewed.get("question"):
                row["question"] = str(reviewed["question"])
    wanted = {value.strip() for value in args.ids.split(",") if value.strip()}
    if wanted:
        rows = [row for row in rows if row["case_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    results: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        expected = row["expected"]
        try:
            response = requests.post(
                args.base_url.rstrip("/") + "/chat",
                headers=headers,
                json={
                    "question": row["question"],
                    "images": [data_url(Path(row["input_path"]))],
                    "session_id": f"{row['case_id']}-{int(time.time())}",
                },
                timeout=args.timeout,
            )
            body = response.json()
            data = body.get("data") if isinstance(body, dict) else {}
            data = data if isinstance(data, dict) else {}
            route = data.get("route") if isinstance(data.get("route"), dict) else {}
            input_images = data.get("input_images") if isinstance(data.get("input_images"), dict) else {}
            verified = (
                input_images.get("verified_grounding")
                if isinstance(input_images.get("verified_grounding"), dict)
                else {}
            )
            answer = str(data.get("answer") or "")
            images = [str(item) for item in (data.get("images") or [])]
            terms = [str(item) for item in (expected.get("rubric_terms") or []) if item]
            result = {
                "case_id": row["case_id"],
                "ok": response.status_code == 200 and body.get("code") == 0,
                "http_status": response.status_code,
                "api_code": body.get("code"),
                "manual_match": str(route.get("manual_id") or "")
                == str(expected.get("manual_id") or ""),
                "image_match": str(expected.get("image_id") or "") in images,
                "rubric_term_match": any(term.lower() in answer.lower() for term in terms),
                "verified_rule_bypassed": not bool(verified.get("matched")),
                "expected": expected,
                "answer": answer,
                "actual_manual_id": route.get("manual_id"),
                "actual_images": images,
                "refusal": data.get("refusal") or {},
                "vision_context": input_images.get("context"),
                "manual_image_matches": input_images.get("manual_image_matches") or {},
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "human_review": "pending",
            }
        except Exception as exc:
            result = {
                "case_id": row["case_id"],
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "human_review": "pending",
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    completed = [row for row in results if row.get("ok")]
    summary = {
        "total": len(results),
        "api_success": len(completed),
        "manual_match": sum(bool(row.get("manual_match")) for row in completed),
        "image_match": sum(bool(row.get("image_match")) for row in completed),
        "rubric_term_match": sum(bool(row.get("rubric_term_match")) for row in completed),
        "verified_rule_bypassed": sum(bool(row.get("verified_rule_bypassed")) for row in completed),
        "human_review": "pending",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
