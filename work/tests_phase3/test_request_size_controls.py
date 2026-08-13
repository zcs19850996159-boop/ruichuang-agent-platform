from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_service_tools.rest_adapter import create_tool_router


class FakeToolService:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, tool_name, payload, *, context=None, stream_callback=None):
        self.calls += 1
        return {
            "schema_version": "1.0",
            "tool": tool_name,
            "request_id": context.request_id,
            "trace_id": context.trace_id,
            "elapsed_ms": 0.0,
            "data": {"answer": "ok"},
        }


def test_tool_api_rejects_body_before_json_expansion() -> None:
    service = FakeToolService()
    app = FastAPI()
    app.include_router(
        create_tool_router(
            service,
            authorize=lambda request: True,
            rate_limiter=lambda request: None,
            timeout_seconds=1.0,
            semaphore=asyncio.Semaphore(1),
            max_body_bytes=128,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/tools/v1/answer_customer_question",
            json={"question": "x" * 256},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "request body too large"
    assert service.calls == 0
