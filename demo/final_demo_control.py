#!/usr/bin/env python3
"""Repeatable preflight, warmup, reset and knowledge rollback for the final demo."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "final_demo_cases.json"
DEFAULT_URL = "http://127.0.0.1:18877"
COMPETITION_KEYCHAIN_SERVICE = "ruichuang-customer-service-api-token"
ENTERPRISE_KEYCHAIN_SERVICE = "ruichuang-final-demo-enterprise-token"


class DemoError(RuntimeError):
    pass


def load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def keychain_token(service: str) -> str:
    environment_name = (
        "RUICHUANG_DEMO_TOKEN"
        if service == COMPETITION_KEYCHAIN_SERVICE
        else "RUICHUANG_DEMO_ENTERPRISE_TOKEN"
    )
    supplied = os.environ.get(environment_name, "").strip()
    if supplied:
        return supplied
    if sys.platform != "darwin":
        raise DemoError(f"{environment_name} is required outside macOS")
    completed = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    token = completed.stdout.strip()
    if completed.returncode or not token:
        raise DemoError(f"macOS 钥匙串中缺少凭据：{service}")
    return token


def request(
    base_url: str,
    path: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json", **(extra_headers or {})}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = json.loads(raw.decode("utf-8")) if raw else {}
        message = (
            detail.get("error", {}).get("message")
            or detail.get("msg")
            or detail.get("detail")
            or f"HTTP {exc.code}"
        )
        raise DemoError(str(message)) from exc
    except urllib.error.URLError as exc:
        raise DemoError(f"无法连接演示服务：{exc.reason}") from exc


def data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def check(base_url: str) -> dict[str, Any]:
    started = time.perf_counter()
    ready_status, ready = request(base_url, "/ready", timeout=15)
    health_status, health = request(base_url, "/health", timeout=15)
    token = keychain_token(COMPETITION_KEYCHAIN_SERVICE)
    tools_status, tools = request(base_url, "/tools/v1", token=token, timeout=15)
    assets = {}
    for relative in ("final_demo_cases.json", "assets/Manual01_18.jpg", "assets/人体工学椅手册.txt"):
        path = ROOT / relative
        if not path.is_file():
            raise DemoError(f"缺少演示资产：{relative}")
        assets[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    ready_data = ready.get("data") or {}
    if ready_status != 200 or ready_data.get("status") != "ready":
        raise DemoError("Phase 3 服务尚未 ready")
    if health_status != 200 or health.get("data", {}).get("status") not in {"ready", "degraded"}:
        raise DemoError("Phase 3 健康检查失败")
    if tools_status != 200:
        raise DemoError("比赛 Token 无法调用 Tool API")
    tools_list = tools.get("tools") or tools.get("data") or []
    if not any(item.get("name") == "answer_customer_question" for item in tools_list):
        raise DemoError("Tool API 未暴露客服核心能力")
    tenant, space, baseline = enterprise_configuration()
    enterprise_token = keychain_token(ENTERPRISE_KEYCHAIN_SERVICE)
    _, identity = request(base_url, "/control/v1/me", token=enterprise_token)
    if (identity.get("data") or {}).get("tenant_id") != tenant:
        raise DemoError("企业演示 Token 与 final-demo 租户不匹配")
    _, active = request(
        base_url,
        (
            f"/control/v1/tenants/{urllib.parse.quote(tenant, safe='')}"
            f"/knowledge-spaces/{urllib.parse.quote(space, safe='')}/active"
        ),
        token=enterprise_token,
    )
    active_version = (active.get("data") or {}).get("version")
    if active_version != baseline:
        raise DemoError(
            f"企业演示知识未复位：预期 {baseline}，实际 {active_version}"
        )
    versions = health.get("data", {}).get("versions", {})
    return {
        "status": "passed",
        "ready": ready_data.get("status"),
        "application_version": (
            versions.get("application_version")
            or versions.get("application")
            or versions.get("service")
        ),
        "tool_count": len(tools_list),
        "enterprise_demo": {
            "tenant_id": tenant,
            "knowledge_space": space,
            "active_version": active_version,
        },
        "assets": assets,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def clear_sessions(base_url: str, token: str, cases: dict[str, Any]) -> list[str]:
    cleared = []
    for case in cases["cases"]:
        session_id = case["session_id"]
        request(
            base_url,
            f"/sessions/{urllib.parse.quote(session_id, safe='')}",
            token=token,
            method="DELETE",
            timeout=15,
        )
        cleared.append(session_id)
    return cleared


def warm_case(base_url: str, token: str, case: dict[str, Any]) -> dict[str, Any]:
    attachments = []
    if case.get("attachment"):
        attachments.append(data_url(ROOT / case["attachment"]))
    started = time.perf_counter()
    _, result = request(
        base_url,
        "/tools/v1/answer_customer_question",
        token=token,
        method="POST",
        timeout=120,
        extra_headers={
            "X-Knowledge-Space-Id": "competition",
            "X-Client-Type": "final-demo-preheat",
        },
        payload={
            "schema_version": "1.0",
            "question": case["question"],
            "attachments": attachments,
            "conversation_context": {"session_id": case["session_id"]},
        },
    )
    data = result.get("data") or {}
    route = data.get("route") or {}
    manual_id = route.get("manual_id")
    images = [
        item if isinstance(item, str) else item.get("image_id")
        for item in data.get("images") or []
    ]
    if manual_id != case["expected_manual"]:
        raise DemoError(f"{case['id']} 手册路由异常：{manual_id}")
    if images != case["expected_images"]:
        raise DemoError(f"{case['id']} 图片集合或顺序异常：{images}")
    if bool(data.get("escalation_required")) != bool(case["expect_escalation"]):
        raise DemoError(f"{case['id']} 转人工状态异常")
    validation = data.get("validation") or data.get("answer_check") or {}
    passed = validation.get("pass", validation.get("ok", True))
    if passed is False:
        raise DemoError(f"{case['id']} 答案校验未通过")
    return {
        "id": case["id"],
        "manual_id": manual_id,
        "images": images,
        "escalation_required": bool(data.get("escalation_required")),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def prepare(base_url: str) -> dict[str, Any]:
    cases = load_cases()
    preflight = check(base_url)
    token = keychain_token(COMPETITION_KEYCHAIN_SERVICE)
    cleared = clear_sessions(base_url, token, cases)
    warmed = [warm_case(base_url, token, case) for case in cases["cases"]]
    return {
        "status": "passed",
        "preflight": preflight,
        "sessions_cleared": cleared,
        "warmed_cases": warmed,
    }


def enterprise_configuration() -> tuple[str, str, str]:
    case = load_cases()["managed_knowledge_case"]
    return case["tenant_id"], case["knowledge_space"], case["baseline_version"]


def rollback_knowledge(base_url: str, target_version: str = "") -> dict[str, Any]:
    tenant, space, baseline = enterprise_configuration()
    target = target_version or baseline
    token = keychain_token(ENTERPRISE_KEYCHAIN_SERVICE)
    encoded_tenant = urllib.parse.quote(tenant, safe="")
    encoded_space = urllib.parse.quote(space, safe="")
    encoded_version = urllib.parse.quote(target, safe="")
    _, active = request(
        base_url,
        f"/control/v1/tenants/{encoded_tenant}/knowledge-spaces/{encoded_space}/active",
        token=token,
    )
    current = (active.get("data") or {}).get("version")
    if current != target:
        request(
            base_url,
            f"/control/v1/tenants/{encoded_tenant}/knowledge-spaces/{encoded_space}/versions/{encoded_version}/rollback",
            token=token,
            method="POST",
            payload={},
        )
    _, verified = request(
        base_url,
        f"/control/v1/tenants/{encoded_tenant}/knowledge-spaces/{encoded_space}/active",
        token=token,
    )
    actual = (verified.get("data") or {}).get("version")
    if actual != target:
        raise DemoError(f"知识回滚验证失败：预期 {target}，实际 {actual}")
    return {
        "status": "passed",
        "tenant_id": tenant,
        "knowledge_space": space,
        "from_version": current,
        "active_version": actual,
    }


def reset(base_url: str) -> dict[str, Any]:
    cases = load_cases()
    token = keychain_token(COMPETITION_KEYCHAIN_SERVICE)
    cleared = clear_sessions(base_url, token, cases)
    return {
        "status": "passed",
        "sessions_cleared": cleared,
        "knowledge": rollback_knowledge(base_url),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "prepare", "reset", "rollback"),
        default="prepare",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RUICHUANG_DEMO_BASE_URL", DEFAULT_URL),
    )
    parser.add_argument("--target-version", default="")
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check(args.base_url)
        elif args.command == "prepare":
            result = prepare(args.base_url)
        elif args.command == "reset":
            result = reset(args.base_url)
        else:
            result = rollback_knowledge(args.base_url, args.target_version)
    except DemoError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
