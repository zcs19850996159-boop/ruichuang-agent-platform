from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ControlPlaneError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or {}

    def public_payload(self) -> dict[str, Any]:
        return {"error_code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    user_id: str
    role: str
    permissions: tuple[str, ...]
    token_id: str
    auth_type: str = "enterprise_token"

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def require(self, permission: str) -> None:
        if not self.can(permission):
            raise ControlPlaneError(
                "permission_denied",
                "permission denied",
                http_status=403,
            )

    def public_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "permissions": list(self.permissions),
            "auth_type": self.auth_type,
        }
