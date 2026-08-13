from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _request_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _assert_ok(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise AssertionError(
            f"{response.request.method} {response.url} returned "
            f"{response.status_code}: {response.text[:1000]}"
        )
    payload = response.json()
    if "error" in payload:
        raise AssertionError(json.dumps(payload["error"], ensure_ascii=False))
    return payload


def run_gate(
    *,
    base_url: str,
    bootstrap_token: str,
    baseline_source: str,
    appended_source: str,
    competition_token: str = "",
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    started = time.perf_counter()
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    tenant_id = f"txt-rehearsal-{run_id}"
    owner_id = f"owner-{run_id}"
    space_id = "official-manuals"
    v1 = "official-txt-v1"
    v2 = "official-txt-v2"

    for source in (baseline_source, appended_source):
        if not Path(source).is_file():
            raise AssertionError(f"source manual does not exist: {source}")

    bootstrap = _assert_ok(
        requests.post(
            f"{base_url}/control/v1/bootstrap",
            headers={
                "Content-Type": "application/json",
                "X-Control-Bootstrap-Token": bootstrap_token,
                "X-Trace-Id": _request_id("txt-bootstrap"),
            },
            json={
                "tenant_id": tenant_id,
                "tenant_name": "Official TXT ingestion rehearsal",
                "owner_user_id": owner_id,
                "owner_display_name": "TXT rehearsal owner",
            },
            timeout=30,
        )
    )["data"]
    enterprise_token = str(bootstrap["api_token"])
    headers = {
        "Authorization": f"Bearer {enterprise_token}",
        "Content-Type": "application/json",
        "X-Knowledge-Space-Id": space_id,
    }
    _assert_ok(
        requests.post(
            f"{base_url}/control/v1/tenants/{tenant_id}/knowledge-spaces",
            headers={**headers, "X-Trace-Id": _request_id("txt-space")},
            json={
                "knowledge_space_id": space_id,
                "name": "Official TXT manuals",
            },
            timeout=30,
        )
    )

    def tool(name: str, body: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        response = requests.post(
            f"{base_url}/tools/v1/{name}",
            headers={**headers, "X-Request-Id": _request_id(f"txt-{name}")},
            json={
                "schema_version": "1.0",
                "knowledge_space_id": space_id,
                **body,
            },
            timeout=timeout,
        )
        return _assert_ok(response)["data"]

    baseline_stage = tool(
        "ingest_customer_manual",
        {
            "source_path": baseline_source,
            "product_id": "air-purifier-official",
        },
    )
    assert baseline_stage["status"] == "staged"
    assert baseline_stage["manifest"]["document"]["parser"] == "official-json-text-v1"
    assert baseline_stage["manifest"]["base_version"] is None
    baseline_quality = tool(
        "evaluate_knowledge_update",
        {
            "staging_id": baseline_stage["staging_id"],
            "regression_report": {
                "status": "passed",
                "total": 3,
                "passed": 3,
                "failed": 0,
            },
        },
    )
    assert baseline_quality["publishable"] is True
    baseline_publish = tool(
        "publish_knowledge_version",
        {
            "staging_id": baseline_stage["staging_id"],
            "version": v1,
            "approved_by": owner_id,
        },
    )
    assert baseline_publish["activation"]["version"] == v1

    appended_stage = tool(
        "ingest_customer_manual",
        {
            "source_path": appended_source,
            "product_id": "ergonomic-chair-official",
        },
    )
    appended_manifest = appended_stage["manifest"]
    assert appended_stage["status"] == "staged"
    assert appended_manifest["document"]["parser"] == "official-json-text-v1"
    assert appended_manifest["base_version"] == v1
    assert appended_manifest["update_mode"] == "append_product"
    assert len(appended_manifest["documents"]) == 2
    appended_quality = tool(
        "evaluate_knowledge_update",
        {
            "staging_id": appended_stage["staging_id"],
            "regression_report": {
                "status": "passed",
                "total": 5,
                "passed": 5,
                "failed": 0,
            },
        },
    )
    assert appended_quality["publishable"] is True
    assert appended_quality["metrics"]["document_count"] == 2
    assert appended_quality["metrics"]["missing_image_count"] == 0
    appended_publish = tool(
        "publish_knowledge_version",
        {
            "staging_id": appended_stage["staging_id"],
            "version": v2,
            "approved_by": owner_id,
        },
    )
    assert appended_publish["activation"]["version"] == v2

    evidence = tool(
        "search_customer_evidence",
        {
            "question": "人体工学椅的腰枕按摩功能如何使用 USB 启用？",
            "top_k": 6,
        },
    )
    assert evidence["version"] == v2
    assert any(
        hit.get("product_id") == "ergonomic-chair-official"
        for hit in evidence["hits"]
    )

    answer = tool(
        "answer_customer_question",
        {
            "question": "人体工学椅的腰枕按摩功能如何启用？",
            "attachments": [],
            "response_mode": "sync",
        },
        timeout=90,
    )
    assert answer["knowledge_version"] == v2
    assert answer["evidence"]
    assert "USB" in answer["answer"].upper()
    assert answer["escalation_required"] is False

    image_response = requests.get(
        (
            f"{base_url}/tools/v1/knowledge-images/{tenant_id}/"
            f"{space_id}/{v2}/Manual02_15"
        ),
        headers=headers,
        timeout=30,
    )
    image_response.raise_for_status()
    assert image_response.content

    rollback_v1 = tool(
        "rollback_knowledge_version",
        {"target_version": v1},
    )
    assert rollback_v1["version"] == v1
    rolled_back_evidence = tool(
        "search_customer_evidence",
        {
            "question": "人体工学椅的腰枕按摩功能如何使用 USB 启用？",
            "top_k": 20,
        },
    )
    assert rolled_back_evidence["version"] == v1
    assert not any(
        hit.get("product_id") == "ergonomic-chair-official"
        for hit in rolled_back_evidence["hits"]
    )
    restore_v2 = tool(
        "rollback_knowledge_version",
        {"target_version": v2},
    )
    assert restore_v2["version"] == v2

    versions = _assert_ok(
        requests.get(
            (
                f"{base_url}/control/v1/tenants/{tenant_id}/"
                f"knowledge-spaces/{space_id}/versions"
            ),
            headers=headers,
            timeout=30,
        )
    )["data"]
    assert {item["version"] for item in versions} == {v1, v2}
    assert next(item for item in versions if item["version"] == v2)["active"] is True

    competition_unchanged = None
    if competition_token:
        competition = _assert_ok(
            requests.post(
                f"{base_url}/chat",
                headers={
                    "Authorization": f"Bearer {competition_token}",
                    "Content-Type": "application/json",
                    "X-Request-Id": _request_id("txt-competition"),
                },
                json={
                    "question": "如何按照手册更换电池？",
                    "images": [],
                    "session_id": _request_id("txt-session"),
                    "stream": False,
                },
                timeout=90,
            )
        )
        assert competition["code"] == 0
        competition_data = competition["data"]
        assert [
            item if isinstance(item, str) else item.get("image_id")
            for item in competition_data["images"]
        ] == ["Manual27_1", "Manual27_2", "Manual27_3"]
        competition_unchanged = True

    return {
        "schema_version": "1.0",
        "gate": "legacy-official-txt-ingestion",
        "status": "passed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "tenant_id": tenant_id,
        "knowledge_space_id": space_id,
        "checks": {
            "official_txt_parser": "official-json-text-v1",
            "baseline_version": v1,
            "appended_version": v2,
            "document_count": appended_quality["metrics"]["document_count"],
            "chunk_count": appended_quality["metrics"]["chunk_count"],
            "image_count": appended_quality["metrics"]["image_count"],
            "missing_image_count": appended_quality["metrics"]["missing_image_count"],
            "managed_answer_contains_usb": True,
            "managed_answer_escalation_required": False,
            "published_image_http_status": image_response.status_code,
            "rollback_to_v1": True,
            "restore_to_v2": True,
            "competition_profile_unchanged": competition_unchanged,
        },
    }


def main() -> None:
    bootstrap_token = os.environ.get("CONTROL_PLANE_BOOTSTRAP_TOKEN", "").strip()
    if not bootstrap_token:
        raise SystemExit("CONTROL_PLANE_BOOTSTRAP_TOKEN is required")
    report = run_gate(
        base_url=os.environ.get(
            "WORKBENCH_BASE_URL",
            "http://127.0.0.1:8878",
        ),
        bootstrap_token=bootstrap_token,
        baseline_source=os.environ.get("TXT_BASELINE_SOURCE", "").strip(),
        appended_source=os.environ.get("TXT_APPENDED_SOURCE", "").strip(),
        competition_token=os.environ.get("KAFU_API_TOKEN", "").strip(),
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
