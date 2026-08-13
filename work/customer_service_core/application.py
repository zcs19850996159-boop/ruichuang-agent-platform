from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .context import RequestContext, RequestContextFactory, bind_request_context
from .errors import CustomerServiceError, ErrorCode, normalize_refusal_payload
from .profiles import CompetitionPatchRegistry


class CustomerServiceRuntime(Protocol):
    """Compatibility seam while legacy behavior is migrated module by module."""

    def answer(
        self,
        payload: dict[str, Any],
        stream_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class CustomerServiceApplication:
    """Stable application service used by REST, MCP and WorkBuddy adapters."""

    legacy_service: CustomerServiceRuntime
    patch_registry: CompetitionPatchRegistry

    def answer(
        self,
        payload: dict[str, Any],
        stream_callback: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CustomerServiceError(
                ErrorCode.INPUT_INVALID,
                "json body must be an object",
                http_status=400,
            )
        request_context = context or RequestContextFactory.from_request(payload=payload)
        profile = self.patch_registry.resolve(request_context.profile)
        if profile.name == "enterprise" and payload.get("_force_competition_patch"):
            raise CustomerServiceError(
                ErrorCode.PERMISSION_DENIED,
                "competition patches cannot be enabled by an enterprise request",
                http_status=403,
            )
        try:
            with bind_request_context(request_context):
                result = self.legacy_service.answer(payload, stream_callback)
        except CustomerServiceError:
            raise
        except ValueError as exc:
            raise CustomerServiceError(
                ErrorCode.INPUT_INVALID,
                str(exc),
                http_status=400,
            ) from exc
        normalize_refusal_payload(result)
        result.setdefault("trace_id", request_context.trace_id)
        result.setdefault("request_context", request_context.public_metadata())
        result.setdefault("profile", self.patch_registry.status(request_context.profile))
        return result

    def __getattr__(self, name: str) -> Any:
        """Compatibility access for health checks and session administration."""
        return getattr(self.legacy_service, name)


def create_application_service() -> CustomerServiceApplication:
    # Imported lazily so API and application code do not depend directly on the
    # competition implementation.
    from .compatibility.legacy_agent import create_legacy_runtime

    return CustomerServiceApplication(
        legacy_service=create_legacy_runtime(),
        patch_registry=CompetitionPatchRegistry(),
    )
