from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from cloud_control.models import ControlPlaneError, Principal
from customer_service_core import RequestContextFactory
from customer_service_core.concurrency import (
    acquire_slot_before,
    release_slot_when_task_finishes,
    run_blocking_with_timeout,
)
from customer_service_core.errors import CustomerServiceError

from .registry import TOOL_DEFINITIONS, list_tools
from .service import CustomerServiceToolService


LOGGER = logging.getLogger(__name__)


async def _read_json_object(
    request: Request,
    *,
    max_body_bytes: int,
) -> dict[str, Any]:
    raw_length = request.headers.get("content-length", "").strip()
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="invalid content-length")
        if declared_length > max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
    body = await request.body()
    if len(body) > max_body_bytes:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="json body must be an object")
    return payload


def create_tool_router(
    service: CustomerServiceToolService,
    *,
    authorize: Callable[[Request], bool],
    rate_limiter: Callable[[Request], None],
    timeout_seconds: float,
    semaphore: asyncio.Semaphore,
    max_body_bytes: int = 16 * 1024 * 1024,
    principal_resolver: Callable[[Request], Principal | None] | None = None,
    enterprise_authorizer: Callable[[Principal, str, str], None] | None = None,
    enterprise_audit: Callable[[Principal, str, str, str], None] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/tools/v1", tags=["Customer Service Tool API v1"])

    def resolve_principal(request: Request) -> Principal | None:
        return principal_resolver(request) if principal_resolver else None

    def request_is_authorized(
        request: Request,
        principal: Principal | None,
    ) -> bool:
        return principal is not None or authorize(request)

    def sse_event(event: str, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"event: {event}\ndata: {body}\n\n"

    def stream_answer(
        payload: dict[str, Any],
        context: Any,
        request_id: str,
    ) -> StreamingResponse:
        async def generate():
            started = time.perf_counter()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + max(0.0, timeout_seconds)
            yield sse_event(
                "status",
                {"stage": "accepted", "request_id": request_id},
            )
            try:
                await acquire_slot_before(semaphore, deadline)
            except asyncio.TimeoutError:
                yield sse_event(
                    "error",
                    {
                        "error_code": "model_timeout",
                        "message": "tool execution timed out while waiting for capacity",
                        "retryable": True,
                        "request_id": request_id,
                    },
                )
                return
            event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
            native_delta_count = 0
            first_delta_ms: float | None = None

            def stream_callback(
                event: str,
                event_payload: dict[str, Any],
            ) -> None:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    (event, dict(event_payload)),
                )

            task = asyncio.create_task(
                asyncio.to_thread(
                    service.execute,
                    "answer_customer_question",
                    payload,
                    context=context,
                    stream_callback=stream_callback,
                )
            )
            ticks = 0
            try:
                while not task.done() or not event_queue.empty():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        yield sse_event(
                            "error",
                            {
                                "error_code": "model_timeout",
                                "message": "tool execution timed out",
                                "retryable": True,
                                "request_id": request_id,
                            },
                        )
                        return
                    try:
                        event, event_payload = await asyncio.wait_for(
                            event_queue.get(),
                            timeout=min(0.75, remaining),
                        )
                        event_payload["request_id"] = request_id
                        if event == "answer_delta":
                            native_delta_count += 1
                            if first_delta_ms is None:
                                first_delta_ms = round(
                                    (time.perf_counter() - started) * 1000,
                                    2,
                                )
                        yield sse_event(event, event_payload)
                    except asyncio.TimeoutError:
                        ticks += 1
                        if not task.done() and ticks % 3 == 0 and event_queue.empty():
                            yield sse_event(
                                "status",
                                {
                                    "stage": "understanding_and_retrieval",
                                    "elapsed_ms": round(
                                        (time.perf_counter() - started) * 1000,
                                        2,
                                    ),
                                    "request_id": request_id,
                                },
                            )

                result = task.result()
                data = result.get("data") or {}
                answer = str(data.get("answer") or "")
                stream_mode = "native_model"
                if native_delta_count == 0:
                    stream_mode = "paced_fallback"
                    yield sse_event(
                        "answer_metadata",
                        {
                            "images": data.get("images") or [],
                            "image_refs": data.get("image_refs") or [],
                            "request_id": request_id,
                        },
                    )
                    chunk_size = max(
                        1,
                        int(os.environ.get("API_FALLBACK_STREAM_CHARS", "2")),
                    )
                    chunk_delay = max(
                        0.0,
                        float(
                            os.environ.get(
                                "API_FALLBACK_STREAM_DELAY",
                                "0.045",
                            )
                        ),
                    )
                    for offset in range(0, len(answer), chunk_size):
                        if first_delta_ms is None:
                            first_delta_ms = round(
                                (time.perf_counter() - started) * 1000,
                                2,
                            )
                        yield sse_event(
                            "answer_delta",
                            {
                                "text": answer[offset : offset + chunk_size],
                                "mode": stream_mode,
                                "request_id": request_id,
                            },
                        )
                        if chunk_delay:
                            await asyncio.sleep(chunk_delay)
                data["streaming"] = {
                    "mode": stream_mode,
                    "native_delta_count": native_delta_count,
                    "first_delta_ms": first_delta_ms,
                }
                yield sse_event("final", result)
            except asyncio.CancelledError:
                raise
            except CustomerServiceError as exc:
                yield sse_event(
                    "error",
                    {
                        **exc.public_payload(),
                        "request_id": request_id,
                    },
                )
            except Exception as exc:
                LOGGER.exception("streaming customer-service tool execution failed")
                detail = (
                    f"{type(exc).__name__}: {exc}"
                    if os.environ.get("API_DEBUG", "0") == "1"
                    else "internal server error"
                )
                yield sse_event(
                    "error",
                    {
                        "error_code": "internal_error",
                        "message": detail,
                        "retryable": False,
                        "request_id": request_id,
                    },
                )
            finally:
                release_slot_when_task_finishes(semaphore, task)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("", summary="List versioned customer-service tools")
    async def tool_list(request: Request) -> dict[str, Any]:
        if not request_is_authorized(request, resolve_principal(request)):
            raise HTTPException(status_code=401, detail="unauthorized")
        return {"schema_version": "1.0", "tools": list_tools("realtime")}

    @router.get(
        "/knowledge-images/{tenant_id}/{space_id}/{version}/{image_id}",
        summary="Read a published managed-knowledge image",
        include_in_schema=False,
    )
    async def knowledge_image(
        tenant_id: str,
        space_id: str,
        version: str,
        image_id: str,
        request: Request,
    ) -> FileResponse:
        principal = resolve_principal(request)
        if not request_is_authorized(request, principal):
            raise HTTPException(status_code=401, detail="unauthorized")
        if principal is not None:
            if not hmac.compare_digest(principal.tenant_id, tenant_id):
                raise HTTPException(status_code=403, detail="permission denied")
            try:
                if enterprise_authorizer:
                    enterprise_authorizer(principal, "knowledge:read", space_id)
                else:
                    principal.require("knowledge:read")
            except ControlPlaneError as exc:
                raise HTTPException(
                    status_code=exc.http_status,
                    detail=exc.message,
                ) from exc
        try:
            path = service.knowledge.resolve_image(
                tenant_id=tenant_id,
                space_id=space_id,
                version=version,
                image_id=image_id,
            )
        except ValueError:
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    @router.post("/{tool_name}", summary="Execute one versioned customer-service tool")
    async def execute_tool(
        tool_name: str,
        request: Request,
    ) -> Any:
        principal = resolve_principal(request)
        if not request_is_authorized(request, principal):
            raise HTTPException(status_code=401, detail="unauthorized")
        rate_limiter(request)
        if tool_name not in TOOL_DEFINITIONS:
            raise HTTPException(status_code=404, detail="unknown tool")
        payload = await _read_json_object(
            request,
            max_body_bytes=max_body_bytes,
        )
        payload.pop("_permissions", None)
        request_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
        context = RequestContextFactory.from_request(payload=payload, headers=request.headers, request_id=request_id)
        is_admin_tool = TOOL_DEFINITIONS[tool_name]["audience"] == "admin"
        if principal is not None:
            permission = str(TOOL_DEFINITIONS[tool_name]["required_permission"])
            try:
                if enterprise_authorizer:
                    enterprise_authorizer(
                        principal,
                        permission,
                        (
                            "competition"
                            if principal.auth_type == "legacy_competition"
                            else context.knowledge_space_id
                        ),
                    )
                else:
                    principal.require(permission)
            except ControlPlaneError as exc:
                return JSONResponse(
                    status_code=exc.http_status,
                    content={
                        "schema_version": "1.0",
                        "error": exc.public_payload(),
                    },
                )
            if principal.auth_type == "legacy_competition":
                context = replace(
                    context,
                    profile="competition",
                    tenant_id="default",
                    knowledge_space_id="competition",
                    user_id=principal.user_id,
                    role=principal.role,
                    permissions=principal.permissions,
                )
            else:
                context = replace(
                    context,
                    profile="enterprise",
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    role=principal.role,
                    permissions=principal.permissions,
                )
        else:
            admin_token = os.environ.get("KNOWLEDGE_ADMIN_TOKEN", "").strip()
            supplied_admin_token = request.headers.get("x-admin-token", "").strip()
            admin_authorized = bool(admin_token) and hmac.compare_digest(supplied_admin_token, admin_token)
            if is_admin_tool and not admin_authorized:
                return JSONResponse(
                    status_code=403,
                    content={
                        "schema_version": "1.0",
                        "error": {
                            "error_code": "permission_denied",
                            "message": "a valid knowledge administration token is required",
                            "retryable": False,
                        },
                    },
                )
            context = replace(
                context,
                role="admin" if admin_authorized else "user",
                permissions=(
                    ("knowledge:write", "knowledge:publish", "audit:read")
                    if admin_authorized
                    else ()
                ),
            )
        if (
            tool_name == "answer_customer_question"
            and str(payload.get("response_mode") or "sync") == "stream"
        ):
            return stream_answer(payload, context, request_id)
        try:
            result = await run_blocking_with_timeout(
                semaphore,
                timeout_seconds,
                service.execute,
                tool_name,
                payload,
                context=context,
            )
            if (
                principal is not None
                and principal.auth_type == "enterprise_token"
                and is_admin_tool
                and enterprise_audit is not None
            ):
                enterprise_audit(
                    principal,
                    tool_name,
                    context.knowledge_space_id,
                    context.trace_id,
                )
            return JSONResponse(status_code=200, content=result)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "schema_version": "1.0",
                    "error": {
                        "error_code": "model_timeout",
                        "message": "tool execution timed out",
                        "retryable": True,
                    },
                },
            )
        except CustomerServiceError as exc:
            return JSONResponse(
                status_code=exc.http_status,
                content={"schema_version": "1.0", "error": exc.public_payload()},
            )
        except Exception as exc:
            LOGGER.exception("customer-service tool execution failed")
            detail = f"{type(exc).__name__}: {exc}" if os.environ.get("API_DEBUG", "0") == "1" else "internal server error"
            return JSONResponse(
                status_code=500,
                content={
                    "schema_version": "1.0",
                    "error": {
                        "error_code": "internal_error",
                        "message": detail,
                        "retryable": False,
                    },
                },
            )

    return router
