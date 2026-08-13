from __future__ import annotations

import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from knowledge_lifecycle import KnowledgeLifecycleService
from knowledge_lifecycle.security import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES

from .models import ControlPlaneError, Principal
from .service import ControlPlaneService


MAX_JSON_BODY_BYTES = int(
    os.environ.get("API_MAX_BODY_BYTES", str(16 * 1024 * 1024))
)


def _trace_id(request: Request) -> str:
    return request.headers.get("x-trace-id", "").strip()


async def _payload(request: Request) -> dict[str, Any]:
    raw_length = request.headers.get("content-length", "").strip()
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise ControlPlaneError(
                "input_invalid",
                "invalid Content-Length",
                http_status=400,
            ) from exc
        if declared_length < 0:
            raise ControlPlaneError(
                "input_invalid",
                "invalid Content-Length",
                http_status=400,
            )
        if declared_length > MAX_JSON_BODY_BYTES:
            raise ControlPlaneError(
                "payload_too_large",
                "request body too large",
                http_status=413,
            )
    body = await request.body()
    if len(body) > MAX_JSON_BODY_BYTES:
        raise ControlPlaneError(
            "payload_too_large",
            "request body too large",
            http_status=413,
        )
    try:
        value = json.loads(body)
    except Exception as exc:
        raise ControlPlaneError("input_invalid", "invalid JSON", http_status=400) from exc
    if not isinstance(value, dict):
        raise ControlPlaneError(
            "input_invalid",
            "json body must be an object",
            http_status=400,
        )
    return value


async def control_error_handler(
    _: Request,
    exc: ControlPlaneError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": exc.public_payload()},
    )


def create_control_router(
    service: ControlPlaneService,
    knowledge: KnowledgeLifecycleService | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/control/v1", tags=["Enterprise control plane v1"])

    def principal(request: Request) -> Principal:
        identity = service.authenticate_authorization(
            request.headers.get("authorization", "")
        )
        if identity is None:
            raise ControlPlaneError("unauthorized", "unauthorized", http_status=401)
        return identity

    def require_knowledge() -> KnowledgeLifecycleService:
        if knowledge is None:
            raise ControlPlaneError(
                "service_unavailable",
                "knowledge version service is unavailable",
                http_status=503,
            )
        return knowledge

    def require_staging_scope(
        identity: Principal,
        *,
        tenant_id: str,
        space_id: str,
        staging_id: str,
        permission: str,
    ) -> dict[str, Any]:
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, permission)
        try:
            metadata = require_knowledge().staging_metadata(staging_id)
        except ValueError as exc:
            raise ControlPlaneError(
                "resource_not_found",
                str(exc),
                http_status=404,
            ) from exc
        if (
            not hmac.compare_digest(str(metadata.get("tenant_id") or ""), identity.tenant_id)
            or not hmac.compare_digest(
                str(metadata.get("knowledge_space_id") or ""),
                space_id,
            )
        ):
            raise ControlPlaneError(
                "resource_not_found",
                "staging package not found",
                http_status=404,
            )
        return metadata

    @router.post("/bootstrap", status_code=201)
    async def bootstrap(request: Request) -> JSONResponse:
        expected = os.environ.get("CONTROL_PLANE_BOOTSTRAP_TOKEN", "").strip()
        supplied = request.headers.get("x-control-bootstrap-token", "").strip()
        if not expected or not hmac.compare_digest(expected, supplied):
            raise ControlPlaneError("unauthorized", "unauthorized", http_status=401)
        payload = await _payload(request)
        result = service.bootstrap_tenant(
            tenant_id=str(payload.get("tenant_id") or ""),
            tenant_name=str(payload.get("tenant_name") or ""),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            owner_display_name=str(payload.get("owner_display_name") or ""),
            trace_id=_trace_id(request),
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.get("/me")
    async def me(request: Request) -> dict[str, Any]:
        return {"data": principal(request).public_dict()}

    @router.post("/tenants/{tenant_id}/members", status_code=201)
    async def create_member(
        tenant_id: str,
        request: Request,
    ) -> JSONResponse:
        identity = principal(request)
        payload = await _payload(request)
        result = service.create_member(
            identity,
            tenant_id=tenant_id,
            user_id=str(payload.get("user_id") or ""),
            display_name=str(payload.get("display_name") or ""),
            role=str(payload.get("role") or ""),
            trace_id=_trace_id(request),
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.get("/tenants/{tenant_id}/members")
    async def list_members(
        tenant_id: str,
        request: Request,
    ) -> dict[str, Any]:
        return {
            "data": service.list_members(
                principal(request),
                tenant_id=tenant_id,
            )
        }

    @router.patch("/tenants/{tenant_id}/members/{user_id}")
    async def update_member(
        tenant_id: str,
        user_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        payload = await _payload(request)
        return {
            "data": service.update_member(
                identity,
                tenant_id=tenant_id,
                user_id=user_id,
                role=str(payload.get("role") or ""),
                status=str(payload.get("status") or ""),
                trace_id=_trace_id(request),
            )
        }

    @router.post("/tenants/{tenant_id}/tokens", status_code=201)
    async def issue_token(
        tenant_id: str,
        request: Request,
    ) -> JSONResponse:
        identity = principal(request)
        payload = await _payload(request)
        result = service.issue_token(
            identity,
            tenant_id=tenant_id,
            user_id=str(payload.get("user_id") or ""),
            expires_at=(
                str(payload["expires_at"]) if payload.get("expires_at") else None
            ),
            trace_id=_trace_id(request),
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.post("/tenants/{tenant_id}/knowledge-spaces", status_code=201)
    async def create_knowledge_space(
        tenant_id: str,
        request: Request,
    ) -> JSONResponse:
        identity = principal(request)
        payload = await _payload(request)
        result = service.create_knowledge_space(
            identity,
            tenant_id=tenant_id,
            space_id=str(payload.get("knowledge_space_id") or ""),
            name=str(payload.get("name") or ""),
            trace_id=_trace_id(request),
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.delete("/tenants/{tenant_id}/tokens/{token_id}")
    async def revoke_token(
        tenant_id: str,
        token_id: str,
        request: Request,
    ) -> dict[str, Any]:
        return {
            "data": service.revoke_token(
                principal(request),
                tenant_id=tenant_id,
                token_id=token_id,
                trace_id=_trace_id(request),
            )
        }

    @router.get("/tenants/{tenant_id}/tokens")
    async def list_tokens(
        tenant_id: str,
        request: Request,
    ) -> dict[str, Any]:
        return {
            "data": service.list_tokens(
                principal(request),
                tenant_id=tenant_id,
            )
        }

    @router.get("/tenants/{tenant_id}/knowledge-spaces")
    async def list_knowledge_spaces(
        tenant_id: str,
        request: Request,
    ) -> dict[str, Any]:
        return {
            "data": service.list_knowledge_spaces(
                principal(request),
                tenant_id=tenant_id,
            )
        }

    @router.get(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/versions"
    )
    async def list_knowledge_versions(
        tenant_id: str,
        space_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:read")
        return {
            "data": require_knowledge().list_versions(
                tenant_id=identity.tenant_id,
                space_id=space_id,
            )
        }

    @router.get(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/active"
    )
    async def active_knowledge_version(
        tenant_id: str,
        space_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:read")
        return {
            "data": require_knowledge().active_version(
                tenant_id=identity.tenant_id,
                space_id=space_id,
            )
        }

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/ingestions",
        status_code=201,
    )
    async def ingest_knowledge_document(
        tenant_id: str,
        space_id: str,
        request: Request,
    ) -> JSONResponse:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:write")
        filename = Path(unquote(request.headers.get("x-filename", ""))).name
        product_id = request.headers.get("x-product-id", "").strip()
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in ALLOWED_EXTENSIONS:
            raise ControlPlaneError(
                "input_invalid",
                "X-Filename must name a supported .txt, .md, .pdf or .docx manual",
                http_status=400,
            )
        if not product_id:
            raise ControlPlaneError(
                "input_invalid",
                "X-Product-Id is required",
                http_status=400,
            )
        try:
            declared_size = int(request.headers.get("content-length", "0") or 0)
        except ValueError as exc:
            raise ControlPlaneError(
                "input_invalid",
                "invalid Content-Length",
                http_status=400,
            ) from exc
        if declared_size > MAX_UPLOAD_BYTES:
            raise ControlPlaneError(
                "payload_too_large",
                f"manual exceeds {MAX_UPLOAD_BYTES} byte limit",
                http_status=413,
            )
        content = await request.body()
        if not content:
            raise ControlPlaneError(
                "input_invalid",
                "manual is empty",
                http_status=400,
            )
        if len(content) > MAX_UPLOAD_BYTES:
            raise ControlPlaneError(
                "payload_too_large",
                f"manual exceeds {MAX_UPLOAD_BYTES} byte limit",
                http_status=413,
            )
        upload_root = require_knowledge().store.root / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="manual-",
                suffix=suffix,
                dir=upload_root,
                delete=False,
            ) as handle:
                handle.write(content)
                temporary = handle.name
            result = require_knowledge().ingest(
                temporary,
                product_id=product_id,
                actor=identity.user_id,
                tenant_id=identity.tenant_id,
                space_id=space_id,
                original_name=filename,
            )
        except (ValueError, RuntimeError) as exc:
            raise ControlPlaneError(
                "knowledge_ingestion_failed",
                str(exc),
                http_status=400,
            ) from exc
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        service.audit_action(
            identity,
            action="knowledge.ingest",
            resource_type="knowledge_staging",
            resource_id=result["staging_id"],
            trace_id=_trace_id(request),
            details={"space_id": space_id, "product_id": product_id},
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/package-ingestions",
        status_code=201,
    )
    async def ingest_product_knowledge_package(
        tenant_id: str,
        space_id: str,
        request: Request,
    ) -> JSONResponse:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:write")
        try:
            declared_size = int(request.headers.get("content-length", "0") or 0)
        except ValueError as exc:
            raise ControlPlaneError(
                "input_invalid",
                "invalid Content-Length",
                http_status=400,
            ) from exc
        if declared_size > MAX_UPLOAD_BYTES:
            raise ControlPlaneError(
                "payload_too_large",
                f"knowledge package exceeds {MAX_UPLOAD_BYTES} byte limit",
                http_status=413,
            )
        content = await request.body()
        if not content:
            raise ControlPlaneError(
                "input_invalid",
                "knowledge package is empty",
                http_status=400,
            )
        if len(content) > MAX_UPLOAD_BYTES:
            raise ControlPlaneError(
                "payload_too_large",
                f"knowledge package exceeds {MAX_UPLOAD_BYTES} byte limit",
                http_status=413,
            )
        upload_root = require_knowledge().store.root / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="product-package-",
                suffix=".zip",
                dir=upload_root,
                delete=False,
            ) as handle:
                handle.write(content)
                temporary = handle.name
            result = require_knowledge().ingest_package(
                temporary,
                actor=identity.user_id,
                tenant_id=identity.tenant_id,
                space_id=space_id,
            )
        except (ValueError, RuntimeError) as exc:
            raise ControlPlaneError(
                "knowledge_package_ingestion_failed",
                str(exc),
                http_status=400,
            ) from exc
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        service.audit_action(
            identity,
            action="knowledge.ingest_package",
            resource_type="knowledge_staging",
            resource_id=result["staging_id"],
            trace_id=_trace_id(request),
            details={
                "space_id": space_id,
                "product_id": result["manifest"]["product_id"],
                "document_count": len(
                    result["manifest"].get("documents") or []
                ),
            },
        )
        return JSONResponse(status_code=201, content={"data": result})

    @router.get(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}"
    )
    async def staging_status(
        tenant_id: str,
        space_id: str,
        staging_id: str,
        request: Request,
    ) -> dict[str, Any]:
        return {
            "data": require_staging_scope(
                principal(request),
                tenant_id=tenant_id,
                space_id=space_id,
                staging_id=staging_id,
                permission="knowledge:read",
            )
        }

    @router.get(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging"
    )
    async def list_staging_packages(
        tenant_id: str,
        space_id: str,
        request: Request,
        limit: int = 100,
    ) -> dict[str, Any]:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:read")
        try:
            rows = require_knowledge().list_staging(
                tenant_id=identity.tenant_id,
                space_id=space_id,
                limit=limit,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "input_invalid",
                str(exc),
                http_status=400,
            ) from exc
        return {"data": rows}

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/evaluate"
    )
    async def evaluate_staging(
        tenant_id: str,
        space_id: str,
        staging_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        require_staging_scope(
            identity,
            tenant_id=tenant_id,
            space_id=space_id,
            staging_id=staging_id,
            permission="knowledge:write",
        )
        payload = await _payload(request)
        regression = payload.get("regression_report")
        if regression is not None and not isinstance(regression, dict):
            raise ControlPlaneError(
                "input_invalid",
                "regression_report must be an object",
                http_status=400,
            )
        try:
            result = require_knowledge().evaluate(
                staging_id,
                regression_report=regression,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "knowledge_evaluation_failed",
                str(exc),
                http_status=400,
            ) from exc
        service.audit_action(
            identity,
            action="knowledge.evaluate",
            resource_type="knowledge_staging",
            resource_id=staging_id,
            trace_id=_trace_id(request),
            details={"publishable": result["publishable"], "space_id": space_id},
        )
        return {"data": result}

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/regression"
    )
    async def run_staging_regression(
        tenant_id: str,
        space_id: str,
        staging_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        require_staging_scope(
            identity,
            tenant_id=tenant_id,
            space_id=space_id,
            staging_id=staging_id,
            permission="knowledge:write",
        )
        payload = await _payload(request)
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ControlPlaneError(
                "input_invalid",
                "cases must be an array",
                http_status=400,
            )
        try:
            result = require_knowledge().run_regression(
                staging_id,
                cases=cases,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "knowledge_regression_failed",
                str(exc),
                http_status=400,
            ) from exc
        service.audit_action(
            identity,
            action="knowledge.regression",
            resource_type="knowledge_staging",
            resource_id=staging_id,
            trace_id=_trace_id(request),
            details={
                "space_id": space_id,
                "total": result["total"],
                "failed": result["failed"],
                "failure_categories": result["failure_categories"],
            },
        )
        return {"data": result}

    @router.get(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/diagnosis"
    )
    async def diagnose_staging(
        tenant_id: str,
        space_id: str,
        staging_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        require_staging_scope(
            identity,
            tenant_id=tenant_id,
            space_id=space_id,
            staging_id=staging_id,
            permission="knowledge:read",
        )
        try:
            result = require_knowledge().diagnose(staging_id)
        except ValueError as exc:
            raise ControlPlaneError(
                "knowledge_diagnosis_failed",
                str(exc),
                http_status=400,
            ) from exc
        return {"data": result}

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/publish"
    )
    async def publish_staging(
        tenant_id: str,
        space_id: str,
        staging_id: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        require_staging_scope(
            identity,
            tenant_id=tenant_id,
            space_id=space_id,
            staging_id=staging_id,
            permission="knowledge:publish",
        )
        payload = await _payload(request)
        version = str(payload.get("version") or "").strip()
        if not version:
            raise ControlPlaneError(
                "input_invalid",
                "version is required",
                http_status=400,
            )
        try:
            result = require_knowledge().publish(
                staging_id,
                tenant_id=identity.tenant_id,
                space_id=space_id,
                version=version,
                approved_by=identity.user_id,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "knowledge_publish_failed",
                str(exc),
                http_status=400,
            ) from exc
        service.audit_action(
            identity,
            action="knowledge.publish",
            resource_type="knowledge_version",
            resource_id=version,
            trace_id=_trace_id(request),
            details={"staging_id": staging_id, "space_id": space_id},
        )
        return {"data": result}

    @router.post(
        "/tenants/{tenant_id}/knowledge-spaces/{space_id}/versions/{version}/rollback"
    )
    async def rollback_version(
        tenant_id: str,
        space_id: str,
        version: str,
        request: Request,
    ) -> dict[str, Any]:
        identity = principal(request)
        service.require_tenant(identity, tenant_id)
        service.require_space(identity, space_id, "knowledge:publish")
        try:
            result = require_knowledge().rollback(
                tenant_id=identity.tenant_id,
                space_id=space_id,
                target_version=version,
                actor=identity.user_id,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "knowledge_rollback_failed",
                str(exc),
                http_status=400,
            ) from exc
        service.audit_action(
            identity,
            action="knowledge.rollback",
            resource_type="knowledge_version",
            resource_id=version,
            trace_id=_trace_id(request),
            details={"space_id": space_id},
        )
        return {"data": result}

    @router.get("/tenants/{tenant_id}/audit")
    async def list_audit(
        tenant_id: str,
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        return {
            "data": service.list_audit(
                principal(request),
                tenant_id=tenant_id,
                limit=limit,
            )
        }

    return router
