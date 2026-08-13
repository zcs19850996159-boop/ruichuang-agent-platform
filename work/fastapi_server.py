from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi

from agent_api import bool_from_env, hybrid_retrieval_status, now_seconds, render_chat_html_v4
from cloud_control import (
    ControlPlaneService,
    Principal,
    create_control_plane_store,
)
from cloud_control.models import ControlPlaneError
from cloud_control.production_config import validate_production_environment
from cloud_control.rest_adapter import create_control_router, control_error_handler
from customer_service_core import RequestContextFactory, create_application_service
from customer_service_core.concurrency import (
    acquire_slot_before,
    release_slot_when_task_finishes,
    run_blocking_with_timeout,
)
from customer_service_core.errors import CustomerServiceError
from customer_service_core.release import ReleaseVersions
from customer_service_tools import create_tool_service
from customer_service_tools.rest_adapter import create_tool_router
from knowledge_lifecycle import KnowledgeLifecycleService, KnowledgeVersionStore


LOGGER = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIGURATION = validate_production_environment()
MANUAL_IMAGE_DIR = (ROOT / "assets" / "manual_images").resolve()
MANUAL_IMAGE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
MANUAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SERVICE = create_application_service()
if (
    PRODUCTION_CONFIGURATION.mode == "production"
    and not SERVICE.state_status().get("ready")
):
    raise RuntimeError("production Redis state store is configured but unavailable")
TOOL_SERVICE = create_tool_service(
    SERVICE,
    knowledge=KnowledgeLifecycleService(KnowledgeVersionStore.from_environment()),
)
RELEASE_VERSIONS = ReleaseVersions.from_environment()
MAX_BODY_BYTES = int(os.environ.get("API_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
REQUEST_TIMEOUT = float(os.environ.get("API_REQUEST_TIMEOUT", "120"))
MAX_CONCURRENCY = max(1, int(os.environ.get("API_MAX_CONCURRENCY", "4")))
RATE_LIMIT = max(1, int(os.environ.get("API_RATE_LIMIT_PER_MINUTE", "60")))
TRUST_PROXY_HEADERS = bool_from_env("API_TRUST_PROXY_HEADERS", False)
TRUSTED_PROXY_IPS = {
    item.strip()
    for item in os.environ.get(
        "API_TRUSTED_PROXY_IPS",
        "127.0.0.1,::1",
    ).split(",")
    if item.strip()
}
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
METRICS = Counter()
METRIC_LOCK = threading.RLock()
RATE_WINDOWS: dict[str, deque[float]] = defaultdict(deque)
LOG_PATH = ROOT / os.environ.get("API_METRICS_LOG", "outputs/api/fastapi_metrics.jsonl")
FEEDBACK_PATH = ROOT / os.environ.get("API_FEEDBACK_LOG", "outputs/api/user_feedback.jsonl")
WARMUP_STATE: dict[str, Any] = {"status": "pending"}
CONTROL_UI_DIR = (Path(__file__).resolve().parent / "cloud_control" / "ui").resolve()
CLOUD_CONTROL_ENABLED = bool_from_env("CLOUD_CONTROL_ENABLED", False)
CONTROL_SERVICE: ControlPlaneService | None = None
if CLOUD_CONTROL_ENABLED:
    CONTROL_SERVICE = ControlPlaneService(
        create_control_plane_store(
            default_sqlite_path=(
                ROOT / "outputs" / "control" / "control-plane.sqlite3"
            ),
        ),
        token_pepper=os.environ.get("CONTROL_PLANE_TOKEN_PEPPER", ""),
    )

app = FastAPI(
    title="Multimodal Customer Agent API",
    version=RELEASE_VERSIONS.application_version,
    description="Production API for multimodal routing, hybrid RAG, traceable sources and confidence-gated answers.",
)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "BearerAuth"
    ] = {"type": "http", "scheme": "bearer"}
    for path, methods in schema.get("paths", {}).items():
        if path in {"/live", "/ready"}:
            continue
        for operation in methods.values():
            if isinstance(operation, dict) and "responses" in operation:
                operation.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("API_CORS_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Request-Id",
        "X-Trace-Id",
        "X-Client-Type",
        "X-Profile",
        "X-Tenant-Id",
        "X-Knowledge-Space-Id",
        "X-Knowledge-Version",
        "X-User-Id",
        "X-User-Role",
        "X-Control-Bootstrap-Token",
        "X-Admin-Token",
        "X-Filename",
        "X-Product-Id",
    ],
)
app.add_exception_handler(ControlPlaneError, control_error_handler)


@app.middleware("http")
async def production_security_headers(
    request: Request,
    call_next: Any,
) -> Any:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if PRODUCTION_CONFIGURATION.mode == "production":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
    return response


@app.on_event("startup")
async def warmup_retrieval() -> None:
    if not bool_from_env("API_STARTUP_WARMUP", True):
        WARMUP_STATE.update({"status": "disabled"})
        return
    try:
        result = await asyncio.to_thread(SERVICE.evidence_builder.warmup)
        WARMUP_STATE.clear()
        WARMUP_STATE.update(result)
    except Exception as exc:
        WARMUP_STATE.clear()
        WARMUP_STATE.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


@app.get("/", include_in_schema=False)
async def home() -> RedirectResponse:
    return RedirectResponse(
        url="/workbench" if CLOUD_CONTROL_ENABLED else "/ui",
        status_code=307,
    )


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui() -> HTMLResponse:
    return HTMLResponse(
        content=render_chat_html_v4(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/admin", include_in_schema=False)
async def control_admin_ui() -> FileResponse:
    if not CLOUD_CONTROL_ENABLED:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        CONTROL_UI_DIR / "admin_ui.html",
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'none'; connect-src 'self'; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


@app.get("/workbench", include_in_schema=False)
async def customer_workbench_ui() -> FileResponse:
    if not CLOUD_CONTROL_ENABLED:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        CONTROL_UI_DIR / "workbench.html",
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data: blob:; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


@app.get("/control-ui/{asset_name}", include_in_schema=False)
async def control_admin_asset(asset_name: str) -> FileResponse:
    allowed_assets = {
        "admin.css",
        "admin.js",
        "workbench.css",
        "workbench.js",
    }
    if not CLOUD_CONTROL_ENABLED or asset_name not in allowed_assets:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        CONTROL_UI_DIR / asset_name,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/manual-images/{image_id}", include_in_schema=False)
async def manual_image(image_id: str, request: Request) -> FileResponse:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    if not MANUAL_IMAGE_ID_RE.fullmatch(image_id):
        raise HTTPException(status_code=404, detail="image not found")
    for extension in MANUAL_IMAGE_EXTENSIONS:
        candidate = (MANUAL_IMAGE_DIR / f"{image_id}{extension}").resolve()
        if candidate.parent == MANUAL_IMAGE_DIR and candidate.is_file():
            return FileResponse(
                candidate,
                headers={"Cache-Control": "private, max-age=3600"},
            )
    raise HTTPException(status_code=404, detail="image not found")


def authorized(request: Request) -> bool:
    token = os.environ.get("KAFU_API_TOKEN", "").strip()
    if not bool_from_env("API_AUTH_REQUIRED", bool(token)):
        return True
    if not token:
        return False
    raw = request.headers.get("authorization", "").strip()
    supplied = raw.split(None, 1)[1].strip() if raw.lower().startswith("bearer ") else raw
    return hmac.compare_digest(supplied, token)


def resolve_tool_principal(request: Request) -> Principal | None:
    if CONTROL_SERVICE is None:
        return None
    identity = CONTROL_SERVICE.authenticate_authorization(
        request.headers.get("authorization", "")
    )
    if identity is not None:
        return identity
    if authorized(request):
        return Principal(
            tenant_id="default",
            user_id="legacy-competition-api",
            role="legacy",
            permissions=("customer:answer", "knowledge:read"),
            token_id="legacy-competition-token",
            auth_type="legacy_competition",
        )
    return None


def authorize_enterprise_tool(
    principal: Principal,
    permission: str,
    space_id: str,
) -> None:
    if principal.auth_type == "legacy_competition":
        principal.require(permission)
        if not hmac.compare_digest(space_id, "competition"):
            raise ControlPlaneError(
                "permission_denied",
                "permission denied",
                http_status=403,
            )
        return
    if CONTROL_SERVICE is None:
        raise ControlPlaneError("unauthorized", "unauthorized", http_status=401)
    CONTROL_SERVICE.require_space(principal, space_id, permission)


def audit_enterprise_tool(
    principal: Principal,
    tool_name: str,
    space_id: str,
    trace_id: str,
) -> None:
    if CONTROL_SERVICE is None:
        return
    CONTROL_SERVICE.store.append_audit(
        tenant_id=principal.tenant_id,
        actor_user_id=principal.user_id,
        action=f"customer_tool.{tool_name}",
        resource_type="knowledge_space",
        resource_id=space_id,
        trace_id=trace_id,
        outcome="success",
        details={"tool": tool_name},
    )


def client_key(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    forwarded = ""
    if TRUST_PROXY_HEADERS and peer in TRUSTED_PROXY_IPS:
        forwarded = (
            request.headers.get("x-forwarded-for", "")
            .split(",", 1)[0]
            .strip()
        )
    host = forwarded or peer
    return hashlib.sha256(host.encode("utf-8")).hexdigest()[:16]


def enforce_rate_limit(request: Request) -> None:
    now = time.monotonic()
    key = client_key(request)
    window = RATE_WINDOWS[key]
    while window and now - window[0] >= 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    window.append(now)


app.include_router(
    create_tool_router(
        TOOL_SERVICE,
        authorize=authorized,
        rate_limiter=enforce_rate_limit,
        timeout_seconds=REQUEST_TIMEOUT,
        semaphore=SEMAPHORE,
        max_body_bytes=MAX_BODY_BYTES,
        principal_resolver=(
            resolve_tool_principal if CLOUD_CONTROL_ENABLED else None
        ),
        enterprise_authorizer=(
            authorize_enterprise_tool if CLOUD_CONTROL_ENABLED else None
        ),
        enterprise_audit=(
            audit_enterprise_tool if CLOUD_CONTROL_ENABLED else None
        ),
    )
)
if CONTROL_SERVICE is not None:
    app.include_router(
        create_control_router(CONTROL_SERVICE, TOOL_SERVICE.knowledge)
    )


def write_metric(event: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with METRIC_LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)


def record_stream_failure(
    *,
    request_id: str,
    context: Any,
    started: float,
    status: int,
    outcome: str,
) -> None:
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    with METRIC_LOCK:
        METRICS["requests_total"] += 1
        METRICS[f"outcome_{outcome}"] += 1
        METRICS["route_unknown"] += 1
        METRICS["elapsed_ms_total"] += int(elapsed_ms)
        METRICS["stream_requests_total"] += 1
    write_metric(
        {
            "timestamp": now_seconds(),
            "request_id": request_id,
            "status": status,
            "outcome": outcome,
            "transport": "sse",
            "elapsed_ms": elapsed_ms,
            "route_type": "",
            "manual_id": "",
            "cache_hit": False,
            "source_count": 0,
            "refusal_type": "",
            "input_image_count": 0,
            "trace_id": context.trace_id,
            "profile": context.profile,
            "tenant_id": context.tenant_id,
            "knowledge_space_id": context.knowledge_space_id,
            "knowledge_version": context.knowledge_version,
        }
    )


async def read_json_object(request: Request) -> dict[str, Any]:
    raw_length = request.headers.get("content-length", "").strip()
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="invalid content-length")
        if declared_length > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="json body must be an object")
    return payload


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"code": exc.status_code, "msg": str(exc.detail), "data": None})


@app.get("/health", summary="Authenticated service and dependency health")
async def health(request: Request) -> dict[str, Any]:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    vision_status = await asyncio.to_thread(SERVICE.vision.runtime_status)
    state_status = SERVICE.state_status()
    service_status = (
        "degraded"
        if WARMUP_STATE.get("status") == "failed"
        or (vision_status.get("enabled") and not vision_status.get("ready"))
        or (state_status.get("configured") and not state_status.get("ready"))
        else "ready"
    )
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "status": service_status,
            "service": "multimodal-customer-agent",
            "framework": "fastapi",
            "chat_endpoint": "/chat",
            "image_input_enabled": bool(vision_status.get("ready")),
            "vision": vision_status,
            "verified_visual_grounding": SERVICE.verified_visual_grounding.status(),
            "visual_retrieval": SERVICE.visual_matcher.status(),
            "remote_media": SERVICE.remote_media.status(),
            "retrieval": hybrid_retrieval_status(),
            "warmup": dict(WARMUP_STATE),
            "state": state_status,
            "versions": RELEASE_VERSIONS.as_dict(),
            "controls": {
                "request_timeout_seconds": REQUEST_TIMEOUT,
                "max_concurrency": MAX_CONCURRENCY,
                "rate_limit_per_minute": RATE_LIMIT,
                "max_body_bytes": MAX_BODY_BYTES,
            },
            "cloud_control": {
                "enabled": CLOUD_CONTROL_ENABLED,
                "api_prefix": "/control/v1" if CLOUD_CONTROL_ENABLED else None,
                "persistence": (
                    CONTROL_SERVICE.health() if CONTROL_SERVICE is not None else None
                ),
            },
            "managed_knowledge": TOOL_SERVICE.knowledge.store.health(),
        },
    }


@app.get("/live", summary="Process liveness")
async def live() -> dict[str, Any]:
    return {
        "code": 0,
        "msg": "success",
        "data": {"status": "alive", "service": "multimodal-customer-agent"},
    }


@app.get("/ready", summary="Required dependency readiness")
async def ready() -> JSONResponse:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    if WARMUP_STATE.get("status") == "failed":
        failures.append("retrieval")
        checks["retrieval"] = {"status": "failed"}
    else:
        checks["retrieval"] = {"status": WARMUP_STATE.get("status", "pending")}
    try:
        checks["managed_knowledge"] = TOOL_SERVICE.knowledge.store.health()
    except Exception:
        failures.append("managed_knowledge")
        checks["managed_knowledge"] = {"status": "failed"}
    if CONTROL_SERVICE is not None:
        try:
            checks["control_plane"] = CONTROL_SERVICE.health()
        except Exception:
            failures.append("control_plane")
            checks["control_plane"] = {"status": "failed"}
    vision_status = await asyncio.to_thread(SERVICE.vision.runtime_status)
    checks["vision"] = {
        key: vision_status.get(key)
        for key in ("enabled", "configured", "model", "ready", "loaded", "status")
    }
    if vision_status.get("enabled") and not vision_status.get("ready"):
        failures.append("vision")
    state_status = SERVICE.state_status()
    checks["state"] = {
        key: state_status.get(key)
        for key in ("backend", "configured", "ready")
    }
    if state_status.get("configured") and not state_status.get("ready"):
        failures.append("state")
    status = "not_ready" if failures else "ready"
    return JSONResponse(
        status_code=503 if failures else 200,
        content={
            "code": 503 if failures else 0,
            "msg": status,
            "data": {
                "status": status,
                "checks": checks,
                "failed_checks": failures,
            },
        },
    )


@app.get("/metrics", summary="Structured runtime counters")
async def metrics(request: Request) -> dict[str, Any]:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    with METRIC_LOCK:
        values = dict(METRICS)
    return {"code": 0, "msg": "success", "data": values}


@app.post("/chat", summary="Text or multimodal customer-service question")
async def chat(request: Request) -> JSONResponse:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    enforce_rate_limit(request)
    payload = await read_json_object(request)
    request_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
    context = RequestContextFactory.from_request(
        payload=payload,
        headers=request.headers,
        request_id=request_id,
    )
    started = time.perf_counter()
    status = 200
    outcome = "success"
    data: dict[str, Any] | None = None
    try:
        data = await run_blocking_with_timeout(
            SEMAPHORE,
            REQUEST_TIMEOUT,
            SERVICE.answer,
            payload,
            context=context,
        )
        data["timestamp"] = now_seconds()
        data["request_id"] = request_id
        client_type = request.headers.get("x-client-type", "").strip()
        if client_type:
            data["client_type"] = client_type
        response = {"code": 0, "msg": "success", "data": data}
    except asyncio.TimeoutError:
        status, outcome = 504, "timeout"
        response = {"code": 504, "msg": "request timed out", "data": None}
    except CustomerServiceError as exc:
        status = exc.http_status
        outcome = exc.code.value
        response = {"code": status, "msg": exc.message, "data": None, "error": exc.public_payload()}
    except Exception as exc:
        LOGGER.exception("synchronous chat request failed")
        status, outcome = 500, "internal_error"
        response = {"code": 500, "msg": "internal server error", "data": None}
        if os.environ.get("API_DEBUG", "0") == "1":
            response["detail"] = f"{type(exc).__name__}: {exc}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    route_type = str(((data or {}).get("route") or {}).get("route_type") or "")
    cache_hit = bool(((data or {}).get("answer_check") or {}).get("cache_hit"))
    with METRIC_LOCK:
        METRICS["requests_total"] += 1
        METRICS[f"outcome_{outcome}"] += 1
        METRICS[f"route_{route_type or 'unknown'}"] += 1
        METRICS["cache_hits_total"] += int(cache_hit)
        METRICS["elapsed_ms_total"] += int(elapsed_ms)
    write_metric({
        "timestamp": now_seconds(),
        "request_id": request_id,
        "status": status,
        "outcome": outcome,
        "elapsed_ms": elapsed_ms,
        "route_type": route_type,
        "manual_id": str(((data or {}).get("route") or {}).get("manual_id") or ""),
        "cache_hit": cache_hit,
        "source_count": len((data or {}).get("sources") or []),
        "refusal_type": str((data or {}).get("refusal_type") or ""),
        "input_image_count": int(((data or {}).get("input_images") or {}).get("received") or 0),
        "trace_id": context.trace_id,
        "profile": context.profile,
        "tenant_id": context.tenant_id,
        "knowledge_space_id": context.knowledge_space_id,
        "knowledge_version": context.knowledge_version,
    })
    return JSONResponse(status_code=status, content=response)


@app.post("/chat/stream", summary="SSE progress and chunked customer-service answer")
async def chat_stream(request: Request) -> StreamingResponse:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    enforce_rate_limit(request)
    payload = await read_json_object(request)
    payload["stream"] = False
    request_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
    client_type = request.headers.get("x-client-type", "").strip()
    context = RequestContextFactory.from_request(
        payload=payload,
        headers=request.headers,
        request_id=request_id,
    )

    async def generate():
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, REQUEST_TIMEOUT)
        metric_recorded = False
        yield sse_event("status", {"stage": "accepted", "request_id": request_id})
        try:
            await acquire_slot_before(SEMAPHORE, deadline)
        except asyncio.TimeoutError:
            record_stream_failure(
                request_id=request_id,
                context=context,
                started=started,
                status=504,
                outcome="timeout",
            )
            yield sse_event(
                "error",
                {
                    "code": 504,
                    "msg": "request timed out while waiting for capacity",
                    "request_id": request_id,
                },
            )
            return
        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        native_delta_count = 0
        first_delta_ms: float | None = None

        def stream_callback(event: str, event_payload: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, (event, dict(event_payload)))

        task = asyncio.create_task(
            asyncio.to_thread(
                SERVICE.answer,
                payload,
                stream_callback,
                context=context,
            )
        )
        ticks = 0
        try:
            while not task.done() or not event_queue.empty():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    record_stream_failure(
                        request_id=request_id,
                        context=context,
                        started=started,
                        status=504,
                        outcome="timeout",
                    )
                    metric_recorded = True
                    yield sse_event("error", {"code": 504, "msg": "request timed out", "request_id": request_id})
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
                            first_delta_ms = round((time.perf_counter() - started) * 1000, 2)
                    yield sse_event(event, event_payload)
                except asyncio.TimeoutError:
                    ticks += 1
                    if not task.done() and ticks % 3 == 0 and event_queue.empty():
                        yield sse_event(
                            "status",
                            {
                                "stage": "understanding_and_retrieval",
                                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                                "request_id": request_id,
                            },
                        )
            data = task.result()
            data["timestamp"] = now_seconds()
            data["request_id"] = request_id
            if client_type:
                data["client_type"] = client_type
            answer = str(data.get("answer") or "")
            stream_mode = "native_model"
            if native_delta_count == 0:
                stream_mode = "paced_fallback"
                chunk_size = max(1, int(os.environ.get("API_FALLBACK_STREAM_CHARS", "2")))
                chunk_delay = max(0.0, float(os.environ.get("API_FALLBACK_STREAM_DELAY", "0.045")))
                for offset in range(0, len(answer), chunk_size):
                    if first_delta_ms is None:
                        first_delta_ms = round((time.perf_counter() - started) * 1000, 2)
                    yield sse_event(
                        "answer_delta",
                        {
                            "text": answer[offset : offset + chunk_size],
                            "request_id": request_id,
                            "mode": stream_mode,
                        },
                    )
                    if chunk_delay:
                        await asyncio.sleep(chunk_delay)
            data["streaming"] = {
                "mode": stream_mode,
                "native_delta_count": native_delta_count,
                "first_delta_ms": first_delta_ms,
            }
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            route_type = str((data.get("route") or {}).get("route_type") or "")
            cache_hit = bool((data.get("answer_check") or {}).get("cache_hit"))
            with METRIC_LOCK:
                METRICS["requests_total"] += 1
                METRICS["outcome_success"] += 1
                METRICS[f"route_{route_type or 'unknown'}"] += 1
                METRICS["cache_hits_total"] += int(cache_hit)
                METRICS["elapsed_ms_total"] += int(elapsed_ms)
                METRICS["stream_requests_total"] += 1
            write_metric(
                {
                    "timestamp": now_seconds(),
                    "request_id": request_id,
                    "status": 200,
                    "outcome": "success",
                    "transport": "sse",
                    "stream_mode": stream_mode,
                    "native_delta_count": native_delta_count,
                    "first_delta_ms": first_delta_ms,
                    "elapsed_ms": elapsed_ms,
                    "route_type": route_type,
                    "manual_id": str((data.get("route") or {}).get("manual_id") or ""),
                    "cache_hit": cache_hit,
                    "source_count": len(data.get("sources") or []),
                    "refusal_type": str(data.get("refusal_type") or ""),
                    "input_image_count": int((data.get("input_images") or {}).get("received") or 0),
                    "trace_id": context.trace_id,
                    "profile": context.profile,
                    "tenant_id": context.tenant_id,
                    "knowledge_space_id": context.knowledge_space_id,
                    "knowledge_version": context.knowledge_version,
                }
            )
            metric_recorded = True
            yield sse_event("final", {"code": 0, "msg": "success", "data": data})
        except asyncio.CancelledError:
            if not metric_recorded:
                record_stream_failure(
                    request_id=request_id,
                    context=context,
                    started=started,
                    status=499,
                    outcome="client_disconnected",
                )
                metric_recorded = True
            raise
        except CustomerServiceError as exc:
            record_stream_failure(
                request_id=request_id,
                context=context,
                started=started,
                status=exc.http_status,
                outcome=exc.code.value,
            )
            metric_recorded = True
            yield sse_event(
                "error",
                {
                    "code": exc.http_status,
                    "msg": exc.message,
                    "error": exc.public_payload(),
                    "request_id": request_id,
                    "trace_id": context.trace_id,
                },
            )
        except Exception as exc:
            LOGGER.exception("streaming chat request failed")
            record_stream_failure(
                request_id=request_id,
                context=context,
                started=started,
                status=500,
                outcome="internal_error",
            )
            metric_recorded = True
            detail = f"{type(exc).__name__}: {exc}" if os.environ.get("API_DEBUG", "0") == "1" else ""
            payload_out = {"code": 500, "msg": "internal server error", "request_id": request_id}
            if detail:
                payload_out["detail"] = detail
            yield sse_event("error", payload_out)
        finally:
            release_slot_when_task_finishes(SEMAPHORE, task)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback", summary="Record user feedback for one answer")
async def feedback(request: Request) -> dict[str, Any]:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await read_json_object(request)
    rating = str(payload.get("rating") or "").strip().lower()
    if rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be up or down")
    event = {
        "timestamp": now_seconds(),
        "feedback_id": uuid.uuid4().hex,
        "rating": rating,
        "request_id": str(payload.get("request_id") or "")[:128],
        "session_id": str(payload.get("session_id") or "")[:128],
        "manual_id": str(payload.get("manual_id") or "")[:64],
        "route_type": str(payload.get("route_type") or "")[:64],
        "comment": str(payload.get("comment") or "")[:1000],
        "client": client_key(request),
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRIC_LOCK:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        METRICS["feedback_total"] += 1
        METRICS[f"feedback_{rating}"] += 1
    return {
        "code": 0,
        "msg": "success",
        "data": {"feedback_id": event["feedback_id"], "rating": rating},
    }


@app.delete("/sessions/{session_id}", summary="Clear one conversation context")
async def clear_session(session_id: str, request: Request) -> dict[str, Any]:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="unauthorized")
    with SERVICE.memory_lock:
        SERVICE.memory.sessions.pop(session_id, None)
        SERVICE.state_store.delete("session", session_id)
    return {"code": 0, "msg": "success", "data": {"session_id": session_id, "cleared": True}}
