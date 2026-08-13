from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


VAGUE_QUESTIONS = (
    "这个部件是什么？",
    "这是什么部件？",
    "这个零件叫什么？",
    "它是什么型号？",
    "这个报错怎么处理？",
    "上面的按钮有什么用？",
    "刚才那个怎么安装？",
    "这里坏了怎么办？",
    "怎么更换？",
    "如何维修？",
    "这玩意怎么用？",
    "帮我看看这个是什么？",
)


def call_chat(url: str, token: str, question: str, session_id: str) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {"question": question, "session_id": session_id, "stream": False},
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def response_data(body: dict) -> dict:
    return body.get("data") or body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    failures: list[dict] = []
    results: list[dict] = []
    run_id = f"reference-gate-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    for index, question in enumerate(VAGUE_QUESTIONS, 1):
        data = response_data(call_chat(args.url, args.token, question, f"{run_id}-{index}"))
        route = data.get("route") or {}
        item = {
            "question": question,
            "route_type": route.get("route_type"),
            "manual_id": route.get("manual_id"),
            "reason": route.get("reason"),
            "image_count": len(data.get("images") or []),
        }
        results.append(item)
        if not (
            item["route_type"] == "clarification"
            and item["manual_id"] == ""
            and item["reason"] == "missing_reference_context"
            and item["image_count"] == 0
        ):
            failures.append(item)

    unclear_session = f"{run_id}-unclear-chain"
    first = response_data(call_chat(args.url, args.token, "这个部件是什么？", unclear_session))
    second = response_data(call_chat(args.url, args.token, "那怎么办？", unclear_session))
    second_route = second.get("route") or {}
    if not (
        (first.get("route") or {}).get("reason") == "missing_reference_context"
        and second_route.get("reason") == "missing_reference_context_followup"
        and not second_route.get("manual_id")
        and not second.get("images")
    ):
        failures.append({"case": "unclear_context_chain", "first": first, "second": second})

    manual_session = f"{run_id}-manual-followup"
    first = response_data(call_chat(args.url, args.token, "空调遥控器没电了，怎么更换电池？", manual_session))
    second = response_data(call_chat(args.url, args.token, "这个部件怎么安装？", manual_session))
    if not (
        (first.get("route") or {}).get("manual_id") == "Manual01"
        and (second.get("route") or {}).get("manual_id") == "Manual01"
    ):
        failures.append({"case": "valid_manual_followup", "first": first, "second": second})

    policy_session = f"{run_id}-policy-followup"
    first = response_data(call_chat(args.url, args.token, "我申请了退款，一般多久到账？", policy_session))
    second = response_data(call_chat(args.url, args.token, "那超过时间怎么办？", policy_session))
    if not (
        (first.get("route") or {}).get("route_type") == "policy_service"
        and (second.get("route") or {}).get("route_type") == "policy_service"
    ):
        failures.append({"case": "valid_policy_followup", "first": first, "second": second})

    summary = {
        "vague_total": len(VAGUE_QUESTIONS),
        "vague_pass": len(VAGUE_QUESTIONS) - sum(1 for item in failures if "question" in item),
        "context_chains": 3,
        "failures": failures,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
