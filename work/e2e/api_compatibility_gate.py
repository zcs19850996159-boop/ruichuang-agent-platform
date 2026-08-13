from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import requests


QUESTION = "如何按照手册更换电池？"
EXPECTED_IMAGES = ["Manual27_1", "Manual27_2", "Manual27_3"]


def _request_id(prefix: str) -> str:
    return f"{prefix}{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _image_ids(data: dict[str, Any]) -> list[str]:
    result = []
    for item in data.get("images") or []:
        image_id = item if isinstance(item, str) else item.get("image_id")
        if image_id:
            result.append(str(image_id))
    return result


def _manual_id(data: dict[str, Any]) -> str:
    route = data.get("route") or {}
    if route.get("manual_id"):
        return str(route["manual_id"])
    selector = (data.get("core_result") or {}).get("selector") or {}
    route = selector.get("route") or {}
    if route.get("manual_id"):
        return str(route["manual_id"])
    for collection in ("sources", "evidence"):
        for item in data.get(collection) or []:
            if item.get("manual_id"):
                return str(item["manual_id"])
    return ""


def _pic_count(answer: str) -> int:
    return str(answer or "").count("<PIC>")


def _parse_sse(response: requests.Response) -> list[dict[str, Any]]:
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    events = []
    event = "message"
    data_lines: list[str] = []
    for raw in response.iter_lines(decode_unicode=True):
        line = str(raw or "")
        if not line:
            if data_lines:
                events.append(
                    {
                        "event": event,
                        "payload": json.loads("\n".join(data_lines)),
                    }
                )
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        events.append(
            {
                "event": event,
                "payload": json.loads("\n".join(data_lines)),
            }
        )
    return events


def _headers(token: str, *, stream: bool = False) -> dict[str, str]:
    return {
        "Accept": "text/event-stream" if stream else "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Client-Type": "phase3-api-e2e-gate",
        "X-Knowledge-Space-Id": "competition",
        "X-Request-Id": _request_id("e2e_req_"),
    }


def _assert_manual27(data: dict[str, Any]) -> None:
    assert _manual_id(data) == "Manual27"
    assert _image_ids(data) == EXPECTED_IMAGES
    assert _pic_count(str(data.get("answer") or "")) == len(EXPECTED_IMAGES)
    if "escalation_required" in data:
        assert data["escalation_required"] is False


def run_gate(
    *,
    base_url: str,
    rollback_url: str,
    token: str,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    rollback_url = rollback_url.rstrip("/")
    started = time.perf_counter()

    ready = requests.get(f"{base_url}/ready", timeout=10)
    ready.raise_for_status()

    stream_response = requests.post(
        f"{base_url}/tools/v1/answer_customer_question",
        headers=_headers(token, stream=True),
        json={
            "schema_version": "1.0",
            "question": QUESTION,
            "attachments": [],
            "conversation_context": {"session_id": _request_id("e2e_stream_")},
            "response_mode": "stream",
        },
        timeout=60,
        stream=True,
    )
    events = _parse_sse(stream_response)
    assert not [item for item in events if item["event"] == "error"]
    final_events = [item for item in events if item["event"] == "final"]
    assert len(final_events) == 1
    final = final_events[0]["payload"]
    deltas = "".join(
        str(item["payload"].get("text") or "")
        for item in events
        if item["event"] == "answer_delta"
    )
    assert deltas == str((final.get("data") or {}).get("answer") or "")
    _assert_manual27(final["data"])

    tool_response = requests.post(
        f"{base_url}/tools/v1/answer_customer_question",
        headers=_headers(token),
        json={
            "schema_version": "1.0",
            "question": QUESTION,
            "attachments": [],
            "conversation_context": {"session_id": _request_id("e2e_sync_")},
            "response_mode": "sync",
        },
        timeout=60,
    )
    tool_response.raise_for_status()
    tool_data = tool_response.json()["data"]
    _assert_manual27(tool_data)

    chat_response = requests.post(
        f"{base_url}/chat",
        headers=_headers(token),
        json={
            "question": QUESTION,
            "images": [],
            "session_id": _request_id("e2e_chat_"),
            "stream": False,
        },
        timeout=60,
    )
    chat_response.raise_for_status()
    chat_payload = chat_response.json()
    assert chat_payload.get("code") == 0
    chat_data = chat_payload["data"]
    _assert_manual27(chat_data)
    assert (chat_data.get("answer_check") or {}).get("constraint_pass") is True

    rollback = None
    for path in ("/ready", "/health"):
        response = requests.get(f"{rollback_url}{path}", timeout=10)
        if response.ok:
            rollback = {"path": path, "status": response.status_code}
            break
    assert rollback is not None

    counts = {}
    for item in events:
        counts[item["event"]] = counts.get(item["event"], 0) + 1
    return {
        "schema_version": "1.0",
        "gate": "phase3-api-compatibility-and-rollback",
        "status": "passed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "checks": {
            "phase3_ready": True,
            "stream_event_counts": counts,
            "streamed_text_equals_final": True,
            "stream_manual_id": _manual_id(final["data"]),
            "stream_images": _image_ids(final["data"]),
            "sync_tool_manual_id": _manual_id(tool_data),
            "sync_tool_images": _image_ids(tool_data),
            "official_chat_manual_id": _manual_id(chat_data),
            "official_chat_images": _image_ids(chat_data),
            "official_chat_constraint_pass": True,
            "rollback": rollback,
        },
    }


def main() -> None:
    token = (
        os.environ.get("CUSTOMER_SERVICE_API_TOKEN")
        or os.environ.get("KAFU_API_TOKEN")
        or ""
    ).strip()
    if not token:
        raise SystemExit("CUSTOMER_SERVICE_API_TOKEN or KAFU_API_TOKEN is required")
    report = run_gate(
        base_url=os.environ.get("WORKBENCH_BASE_URL", "http://127.0.0.1:8878"),
        rollback_url=os.environ.get("ROLLBACK_BASE_URL", "http://127.0.0.1:8877"),
        token=token,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
