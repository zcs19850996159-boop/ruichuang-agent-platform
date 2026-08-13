from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from .models import ControlPlaneError, Principal
from .store import LAST_OWNER_ERROR, safe_id
from .store_contract import ControlPlaneStore


ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "owner": (
        "tenant:manage",
        "users:manage",
        "tokens:manage",
        "customer:answer",
        "knowledge:read",
        "knowledge:write",
        "knowledge:publish",
        "audit:read",
    ),
    "admin": (
        "users:manage",
        "tokens:manage",
        "customer:answer",
        "knowledge:read",
        "knowledge:write",
        "knowledge:publish",
        "audit:read",
    ),
    "knowledge_manager": (
        "customer:answer",
        "knowledge:read",
        "knowledge:write",
        "knowledge:publish",
        "audit:read",
    ),
    "agent": ("customer:answer", "knowledge:read"),
    "viewer": ("knowledge:read", "audit:read"),
}


class ControlPlaneService:
    TOKEN_PREFIX = "rcp_"

    def __init__(self, store: ControlPlaneStore, *, token_pepper: str) -> None:
        if len(token_pepper.encode("utf-8")) < 32:
            raise ValueError("control-plane token pepper must be at least 32 bytes")
        self.store = store
        self._pepper = token_pepper.encode("utf-8")

    def health(self) -> dict[str, Any]:
        return self.store.health()

    def _hash_token(self, token: str) -> str:
        return hmac.new(self._pepper, token.encode("utf-8"), hashlib.sha256).hexdigest()

    def _new_token(self) -> tuple[str, str, str, str]:
        token_id = f"tok-{uuid.uuid4().hex}"
        token = f"{self.TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
        return token_id, token, self._hash_token(token), token[:12]

    @staticmethod
    def _role(role: str) -> str:
        normalized = str(role or "").strip()
        if normalized not in ROLE_PERMISSIONS:
            raise ControlPlaneError(
                "input_invalid",
                "invalid role",
                http_status=400,
            )
        return normalized

    @staticmethod
    def _trace_id(trace_id: str | None) -> str:
        return str(trace_id or f"trace_{uuid.uuid4().hex}")[:128]

    def bootstrap_tenant(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        owner_user_id: str,
        owner_display_name: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        if not tenant_name.strip() or not owner_display_name.strip():
            raise ControlPlaneError(
                "input_invalid",
                "tenant and owner names are required",
                http_status=400,
            )
        token_id, token, token_hash, token_prefix = self._new_token()
        try:
            self.store.create_tenant_owner(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                user_id=owner_user_id,
                display_name=owner_display_name,
                role="owner",
                token_id=token_id,
                token_hash=token_hash,
                token_prefix=token_prefix,
            )
        except ValueError as exc:
            raise ControlPlaneError(
                "input_invalid",
                str(exc),
                http_status=400,
            ) from exc
        except Exception as exc:
            raise ControlPlaneError(
                "resource_conflict",
                "tenant or owner already exists",
                http_status=409,
            ) from exc
        self.store.append_audit(
            tenant_id=tenant_id,
            actor_user_id=owner_user_id,
            action="tenant.bootstrap",
            resource_type="tenant",
            resource_id=tenant_id,
            trace_id=self._trace_id(trace_id),
            outcome="success",
        )
        return {
            "tenant_id": safe_id(tenant_id, "tenant_id"),
            "owner_user_id": safe_id(owner_user_id, "user_id"),
            "role": "owner",
            "api_token": token,
            "token_id": token_id,
            "token_returned_once": True,
        }

    def authenticate(self, token: str) -> Principal | None:
        if not token.startswith(self.TOKEN_PREFIX):
            return None
        record = self.store.identity_for_hash(self._hash_token(token))
        if record is None:
            return None
        if (
            record["status"] != "active"
            or record["tenant_status"] != "active"
            or record["user_status"] != "active"
            or record["membership_status"] != "active"
        ):
            return None
        expires_at = record.get("expires_at")
        if expires_at and datetime.fromisoformat(expires_at) <= datetime.now(UTC):
            return None
        role = str(record["role"])
        permissions = ROLE_PERMISSIONS.get(role)
        if permissions is None:
            return None
        return Principal(
            tenant_id=str(record["tenant_id"]),
            user_id=str(record["user_id"]),
            role=role,
            permissions=permissions,
            token_id=str(record["token_id"]),
        )

    def authenticate_authorization(self, authorization: str) -> Principal | None:
        raw = str(authorization or "").strip()
        if not raw.lower().startswith("bearer "):
            return None
        return self.authenticate(raw.split(None, 1)[1].strip())

    @staticmethod
    def require_tenant(principal: Principal, tenant_id: str) -> None:
        try:
            normalized = safe_id(tenant_id, "tenant_id")
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        if not hmac.compare_digest(principal.tenant_id, normalized):
            raise ControlPlaneError(
                "permission_denied",
                "permission denied",
                http_status=403,
            )

    def create_member(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        user_id: str,
        display_name: str,
        role: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_tenant(principal, tenant_id)
        principal.require("users:manage")
        role = self._role(role)
        if role == "owner" and principal.role != "owner":
            raise ControlPlaneError("permission_denied", "permission denied", http_status=403)
        if not display_name.strip():
            raise ControlPlaneError(
                "input_invalid",
                "display_name is required",
                http_status=400,
            )
        try:
            result = self.store.add_membership(
                tenant_id=principal.tenant_id,
                user_id=user_id,
                display_name=display_name,
                role=role,
            )
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        except Exception as exc:
            raise ControlPlaneError(
                "resource_conflict",
                "membership already exists",
                http_status=409,
            ) from exc
        self._audit(
            principal,
            action="membership.create",
            resource_type="user",
            resource_id=user_id,
            trace_id=trace_id,
            details={"role": role},
        )
        return result

    def list_members(
        self,
        principal: Principal,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        self.require_tenant(principal, tenant_id)
        principal.require("users:manage")
        return self.store.list_members(principal.tenant_id)

    def update_member(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        user_id: str,
        role: str,
        status: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_tenant(principal, tenant_id)
        principal.require("users:manage")
        role = self._role(role)
        normalized_status = str(status or "").strip()
        if normalized_status not in {"active", "disabled"}:
            raise ControlPlaneError(
                "input_invalid",
                "invalid membership status",
                http_status=400,
            )
        try:
            current = self.store.membership(principal.tenant_id, user_id)
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        if current is None:
            raise ControlPlaneError(
                "resource_not_found",
                "membership not found",
                http_status=404,
            )
        if (
            current["role"] == "owner" or role == "owner"
        ) and principal.role != "owner":
            raise ControlPlaneError("permission_denied", "permission denied", http_status=403)
        if (
            user_id == principal.user_id
            and normalized_status == "disabled"
        ):
            raise ControlPlaneError(
                "self_disable_not_allowed",
                "use another owner or administrator to disable this membership",
                http_status=409,
            )
        try:
            result = self.store.update_membership(
                tenant_id=principal.tenant_id,
                user_id=user_id,
                role=role,
                status=normalized_status,
            )
        except ValueError as exc:
            if str(exc) == LAST_OWNER_ERROR:
                raise ControlPlaneError(
                    "last_owner_required",
                    LAST_OWNER_ERROR,
                    http_status=409,
                ) from exc
            raise ControlPlaneError(
                "input_invalid",
                str(exc),
                http_status=400,
            ) from exc
        self._audit(
            principal,
            action="membership.update",
            resource_type="user",
            resource_id=user_id,
            trace_id=trace_id,
            details={"role": role, "status": normalized_status},
        )
        return result

    def issue_token(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        user_id: str,
        expires_at: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_tenant(principal, tenant_id)
        principal.require("tokens:manage")
        if expires_at:
            try:
                expiration = datetime.fromisoformat(expires_at)
            except ValueError as exc:
                raise ControlPlaneError(
                    "input_invalid",
                    "expires_at must be ISO-8601",
                    http_status=400,
                ) from exc
            if expiration.tzinfo is None or expiration <= datetime.now(UTC):
                raise ControlPlaneError(
                    "input_invalid",
                    "expires_at must be a future timezone-aware timestamp",
                    http_status=400,
                )
        token_id, token, token_hash, token_prefix = self._new_token()
        try:
            metadata = self.store.insert_token(
                token_id=token_id,
                tenant_id=principal.tenant_id,
                user_id=user_id,
                token_hash=token_hash,
                token_prefix=token_prefix,
                expires_at=expires_at,
            )
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        self._audit(
            principal,
            action="token.issue",
            resource_type="api_token",
            resource_id=token_id,
            trace_id=trace_id,
            details={"for_user_id": user_id, "expires_at": expires_at},
        )
        return {**metadata, "api_token": token, "token_returned_once": True}

    def create_knowledge_space(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        space_id: str,
        name: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_tenant(principal, tenant_id)
        principal.require("knowledge:write")
        if not name.strip():
            raise ControlPlaneError("input_invalid", "name is required", http_status=400)
        try:
            result = self.store.create_knowledge_space(
                tenant_id=principal.tenant_id,
                space_id=space_id,
                name=name,
                created_by=principal.user_id,
            )
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        except Exception as exc:
            raise ControlPlaneError(
                "resource_conflict",
                "knowledge space already exists",
                http_status=409,
            ) from exc
        self._audit(
            principal,
            action="knowledge_space.create",
            resource_type="knowledge_space",
            resource_id=space_id,
            trace_id=trace_id,
        )
        return result

    def revoke_token(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        token_id: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_tenant(principal, tenant_id)
        principal.require("tokens:manage")
        try:
            result = self.store.revoke_token(
                tenant_id=principal.tenant_id,
                token_id=token_id,
            )
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        self._audit(
            principal,
            action="token.revoke",
            resource_type="api_token",
            resource_id=token_id,
            trace_id=trace_id,
            details={"for_user_id": result["user_id"]},
        )
        return result

    def list_tokens(
        self,
        principal: Principal,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        self.require_tenant(principal, tenant_id)
        principal.require("tokens:manage")
        return self.store.list_tokens(principal.tenant_id)

    def list_knowledge_spaces(
        self,
        principal: Principal,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        self.require_tenant(principal, tenant_id)
        principal.require("knowledge:read")
        return self.store.list_knowledge_spaces(principal.tenant_id)

    def require_space(
        self,
        principal: Principal,
        space_id: str,
        permission: str,
    ) -> dict[str, Any]:
        principal.require(permission)
        try:
            space = self.store.knowledge_space(principal.tenant_id, space_id)
        except ValueError as exc:
            raise ControlPlaneError("input_invalid", str(exc), http_status=400) from exc
        if space is None or space["status"] != "active":
            raise ControlPlaneError(
                "knowledge_space_not_found",
                "knowledge space not found",
                http_status=404,
            )
        return space

    def list_audit(
        self,
        principal: Principal,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self.require_tenant(principal, tenant_id)
        principal.require("audit:read")
        return self.store.list_audit(principal.tenant_id, limit=limit)

    def audit_action(
        self,
        principal: Principal,
        *,
        action: str,
        resource_type: str,
        resource_id: str,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit(
            principal,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            trace_id=trace_id,
            details=details,
        )

    def _audit(
        self,
        principal: Principal,
        *,
        action: str,
        resource_type: str,
        resource_id: str,
        trace_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.store.append_audit(
            tenant_id=principal.tenant_id,
            actor_user_id=principal.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            trace_id=self._trace_id(trace_id),
            outcome="success",
            details=details,
        )
