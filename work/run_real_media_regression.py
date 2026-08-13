from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]


def check_case(case: dict[str, Any], data: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    input_images = data.get("input_images") if isinstance(data.get("input_images"), dict) else {}
    grounding = (
        input_images.get("verified_grounding")
        if isinstance(input_images.get("verified_grounding"), dict)
        else {}
    )
    answer = str(data.get("answer") or "")
    images = [str(x) for x in (data.get("images") or [])]
    remote_media = data.get("remote_media") if isinstance(data.get("remote_media"), dict) else {}
    if str(route.get("manual_id") or "") != str(case.get("expected_manual") or ""):
        issues.append(f"manual={route.get('manual_id')!r}")
    if str(grounding.get("rule_id") or "") != str(case.get("expected_rule") or ""):
        issues.append(f"grounding_rule={grounding.get('rule_id')!r}")
    expected_images = [str(x) for x in (case.get("expected_images") or [])]
    if images != expected_images:
        issues.append(f"images={images!r}")
    must_any = [str(x) for x in (case.get("answer_must_contain_any") or [])]
    if must_any and not any(term in answer for term in must_any):
        issues.append(f"answer_missing_any={must_any!r}")
    forbidden = [str(x) for x in (case.get("answer_must_not_contain") or [])]
    hit = [term for term in forbidden if term in answer]
    if hit:
        issues.append(f"answer_forbidden={hit!r}")
    if answer.count("<PIC>") != len(images):
        issues.append(f"pic_image_mismatch={answer.count('<PIC>')}:{len(images)}")
    expected_media_kind = str(case.get("expected_media_kind") or "")
    if expected_media_kind:
        media_items = [
            item
            for item in (remote_media.get("items") or [])
            if isinstance(item, dict) and str(item.get("media_kind") or "") == expected_media_kind
        ]
        if not media_items:
            issues.append(f"media_kind_missing={expected_media_kind!r}")
        expected_frame_count = int(case.get("expected_frame_count") or 0)
        minimum_frame_count = int(case.get("minimum_frame_count") or 0)
        maximum_frame_count = int(case.get("maximum_frame_count") or 0)
        if expected_frame_count and len(media_items) != expected_frame_count:
            issues.append(f"frame_count={len(media_items)}")
        if minimum_frame_count and len(media_items) < minimum_frame_count:
            issues.append(f"frame_count_below_minimum={len(media_items)}")
        if maximum_frame_count and len(media_items) > maximum_frame_count:
            issues.append(f"frame_count_above_maximum={len(media_items)}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(ROOT / "assets" / "real_media_regression_cases.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token", default=os.environ.get("KAFU_API_TOKEN", ""))
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "rag_agent" / "real_media_regression.jsonl"),
    )
    parser.add_argument("--ids", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("id")) in wanted]
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            response = requests.post(
                args.base_url.rstrip("/") + "/chat",
                headers=headers,
                json={
                    "question": case["question"],
                    "session_id": f"real-media-regression-{case['id']}-{int(time.time())}",
                },
                timeout=args.timeout,
            )
            body = response.json()
            data = body.get("data") if isinstance(body, dict) else {}
            data = data if isinstance(data, dict) else {}
            issues = check_case(case, data)
            result = {
                "id": case["id"],
                "ok": response.status_code == 200 and body.get("code") == 0 and not issues,
                "http_status": response.status_code,
                "api_code": body.get("code"),
                "issues": issues,
                "manual_id": (data.get("route") or {}).get("manual_id"),
                "answer": data.get("answer"),
                "images": data.get("images") or [],
                "vision_context": (data.get("input_images") or {}).get("context"),
                "verified_grounding": (data.get("input_images") or {}).get("verified_grounding"),
                "remote_media": data.get("remote_media") or {},
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            result = {
                "id": case["id"],
                "ok": False,
                "http_status": 0,
                "api_code": None,
                "issues": [f"{type(exc).__name__}: {exc}"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    passed = sum(bool(row.get("ok")) for row in results)
    summary = {"total": len(results), "passed": passed, "failed": len(results) - passed}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
