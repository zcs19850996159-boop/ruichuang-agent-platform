from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloud_control.models import ControlPlaneError, Principal
from customer_service_tools.rest_adapter import create_tool_router


class FakeToolService:
    def __init__(self) -> None:
        self.contexts = []

    def execute(self, tool_name, payload, context=None, stream_callback=None):
        self.contexts.append(context)
        if stream_callback is not None:
            stream_callback("answer_delta", {"text": "o"})
            stream_callback("answer_delta", {"text": "k"})
        return {
            "schema_version": "1.0",
            "tool": tool_name,
            "request_id": context.request_id,
            "trace_id": context.trace_id,
            "elapsed_ms": 0.1,
            "data": {"answer": "ok"},
        }


def principal(role: str = "agent") -> Principal:
    permissions = (
        ("customer:answer", "knowledge:read")
        if role == "agent"
        else (
            "customer:answer",
            "knowledge:read",
            "knowledge:write",
            "knowledge:publish",
            "audit:read",
        )
    )
    return Principal(
        tenant_id="tenant-a",
        user_id="alice",
        role=role,
        permissions=permissions,
        token_id="tok-test",
    )


def competition_principal() -> Principal:
    return Principal(
        tenant_id="default",
        user_id="legacy-competition-api",
        role="legacy",
        permissions=("customer:answer", "knowledge:read"),
        token_id="legacy-competition-token",
        auth_type="legacy_competition",
    )


def app_for(identity: Principal):
    fake = FakeToolService()
    app = FastAPI()

    def authorize_enterprise(
        resolved: Principal,
        permission: str,
        space_id: str,
    ) -> None:
        resolved.require(permission)
        if resolved.auth_type == "legacy_competition" and space_id != "competition":
            raise ControlPlaneError(
                "permission_denied",
                "permission denied",
                http_status=403,
            )
        if resolved.auth_type != "legacy_competition" and space_id != "manuals":
            raise ControlPlaneError(
                "knowledge_space_not_found",
                "knowledge space not found",
                http_status=404,
            )

    app.include_router(
        create_tool_router(
            fake,
            authorize=lambda request: False,
            rate_limiter=lambda request: None,
            timeout_seconds=10,
            semaphore=asyncio.Semaphore(2),
            principal_resolver=lambda request: identity,
            enterprise_authorizer=authorize_enterprise,
        )
    )
    return app, fake


def test_untrusted_identity_headers_are_overwritten() -> None:
    app, fake = app_for(principal())
    response = TestClient(app).post(
        "/tools/v1/answer_customer_question",
        headers={
            "X-Tenant-Id": "tenant-b",
            "X-User-Id": "mallory",
            "X-User-Role": "owner",
            "X-Knowledge-Space-Id": "manuals",
        },
        json={"question": "How do I start it?"},
    )
    assert response.status_code == 200
    context = fake.contexts[-1]
    assert context.tenant_id == "tenant-a"
    assert context.user_id == "alice"
    assert context.role == "agent"
    assert context.profile == "enterprise"


def test_agent_cannot_execute_admin_tool() -> None:
    app, fake = app_for(principal())
    response = TestClient(app).post(
        "/tools/v1/publish_knowledge_version",
        headers={"X-Knowledge-Space-Id": "manuals"},
        json={
            "staging_id": "stg-test",
            "version": "v1",
            "approved_by": "alice",
        },
    )
    assert response.status_code == 403
    assert not fake.contexts


def test_unknown_space_is_rejected_before_core_call() -> None:
    app, fake = app_for(principal())
    response = TestClient(app).post(
        "/tools/v1/answer_customer_question",
        headers={"X-Knowledge-Space-Id": "tenant-b-space"},
        json={"question": "How do I start it?"},
    )
    assert response.status_code == 404
    assert not fake.contexts


def test_competition_token_forces_frozen_profile_and_space() -> None:
    app, fake = app_for(competition_principal())
    response = TestClient(app).post(
        "/tools/v1/answer_customer_question",
        headers={
            "X-Tenant-Id": "tenant-b",
            "X-User-Id": "mallory",
            "X-User-Role": "owner",
            "X-Knowledge-Space-Id": "enterprise-space",
        },
        json={"question": "如何按照手册更换电池？"},
    )
    assert response.status_code == 200
    context = fake.contexts[-1]
    assert context.profile == "competition"
    assert context.tenant_id == "default"
    assert context.knowledge_space_id == "competition"
    assert context.user_id == "legacy-competition-api"
    assert context.role == "legacy"


def test_streaming_tool_keeps_one_call_and_frozen_competition_context() -> None:
    app, fake = app_for(competition_principal())
    with TestClient(app).stream(
        "POST",
        "/tools/v1/answer_customer_question",
        headers={
            "X-Tenant-Id": "tenant-b",
            "X-User-Role": "owner",
            "X-Knowledge-Space-Id": "enterprise-space",
        },
        json={
            "question": "如何按照手册更换电池？",
            "response_mode": "stream",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body.count("event: answer_delta") == 2
    assert "event: final" in body
    assert len(fake.contexts) == 1
    context = fake.contexts[0]
    assert context.profile == "competition"
    assert context.tenant_id == "default"
    assert context.knowledge_space_id == "competition"


def test_competition_token_cannot_execute_admin_tool() -> None:
    app, fake = app_for(competition_principal())
    response = TestClient(app).post(
        "/tools/v1/publish_knowledge_version",
        headers={"X-Knowledge-Space-Id": "competition"},
        json={
            "staging_id": "stg-test",
            "version": "v1",
            "approved_by": "legacy",
        },
    )
    assert response.status_code == 403
    assert not fake.contexts
