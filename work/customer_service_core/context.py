from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Request-scoped identity and version information.

    The competition API supplies safe defaults. Enterprise adapters can fill
    these fields from authenticated headers without changing the core flow.
    """

    request_id: str
    trace_id: str
    profile: str
    tenant_id: str
    knowledge_space_id: str
    knowledge_version: str
    application_version: str
    model_version: str
    prompt_version: str
    user_id: str | None = None
    role: str | None = None
    permissions: tuple[str, ...] = ()

    def public_metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "profile": self.profile,
            "tenant_id": self.tenant_id,
            "knowledge_space_id": self.knowledge_space_id,
            "knowledge_version": self.knowledge_version,
            "application_version": self.application_version,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
        }

    def cache_namespace(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "knowledge_space_id": self.knowledge_space_id,
            "knowledge_version": self.knowledge_version,
            "profile": self.profile,
            "application_version": self.application_version,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "permission_scope": hashlib.sha256(
                json.dumps(sorted(self.permissions), separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16],
        }


_DEFAULT_CONTEXT = RequestContext(
    request_id="legacy",
    trace_id="legacy",
    profile="competition",
    tenant_id="default",
    knowledge_space_id="competition",
    knowledge_version=os.environ.get("KNOWLEDGE_VERSION", "competition-kb-v1"),
    application_version=os.environ.get("APPLICATION_VERSION", "3.5.0-phase3c"),
    model_version=os.environ.get("MODEL_CONFIGURATION_VERSION", "model-config-v1"),
    prompt_version=os.environ.get("PROMPT_VERSION", "prompt-v1"),
)
_CURRENT_CONTEXT: contextvars.ContextVar[RequestContext] = contextvars.ContextVar(
    "customer_service_request_context",
    default=_DEFAULT_CONTEXT,
)


def current_request_context() -> RequestContext:
    return _CURRENT_CONTEXT.get()


@contextlib.contextmanager
def bind_request_context(context: RequestContext) -> Iterator[RequestContext]:
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


class RequestContextFactory:
    """Build a validated context from an HTTP/tool adapter request."""

    MAX_FIELD_LENGTH = 128

    @classmethod
    def _value(
        cls,
        headers: Mapping[str, str],
        name: str,
        default: str,
    ) -> str:
        value = str(headers.get(name, "") or default).strip()
        return value[: cls.MAX_FIELD_LENGTH] or default

    @classmethod
    def from_request(
        cls,
        *,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        request_id: str | None = None,
    ) -> RequestContext:
        normalized_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        effective_request_id = (
            request_id
            or normalized_headers.get("x-request-id")
            or str(payload.get("request_id") or "")
            or uuid.uuid4().hex
        )[: cls.MAX_FIELD_LENGTH]
        trace_id = (
            normalized_headers.get("x-trace-id")
            or str(payload.get("trace_id") or "")
            or f"trace_{uuid.uuid4().hex}"
        )[: cls.MAX_FIELD_LENGTH]
        permissions_raw = payload.get("_permissions") or ()
        permissions = tuple(
            str(item)[: cls.MAX_FIELD_LENGTH]
            for item in permissions_raw
            if isinstance(item, (str, int))
        )
        return RequestContext(
            request_id=effective_request_id,
            trace_id=trace_id,
            profile=cls._value(
                normalized_headers,
                "x-profile",
                os.environ.get("CUSTOMER_SERVICE_PROFILE", "competition"),
            ),
            tenant_id=cls._value(normalized_headers, "x-tenant-id", "default"),
            knowledge_space_id=cls._value(
                normalized_headers,
                "x-knowledge-space-id",
                os.environ.get("KNOWLEDGE_SPACE_ID", "competition"),
            ),
            knowledge_version=cls._value(
                normalized_headers,
                "x-knowledge-version",
                os.environ.get("KNOWLEDGE_VERSION", "competition-kb-v1"),
            ),
            application_version=os.environ.get("APPLICATION_VERSION", "3.5.0-phase3c"),
            model_version=os.environ.get("MODEL_CONFIGURATION_VERSION", "model-config-v1"),
            prompt_version=os.environ.get("PROMPT_VERSION", "prompt-v1"),
            user_id=cls._value(normalized_headers, "x-user-id", "") or None,
            role=cls._value(normalized_headers, "x-user-role", "") or None,
            permissions=permissions,
        )
