from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .migrations import apply_sqlite_migrations


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
LAST_OWNER_ERROR = "the final active owner cannot be demoted or disabled"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_id(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_ID.fullmatch(normalized):
        raise ValueError(f"invalid {field}")
    return normalized


class SQLiteControlPlaneStore:
    """SQLite development and single-node acceptance implementation."""

    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                self.schema_version = apply_sqlite_migrations(connection)
                connection.execute("PRAGMA journal_mode = WAL")
                connection.commit()
            finally:
                connection.close()

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
        return {
            "status": "ready",
            "backend": self.backend,
            "schema_version": int(row["value"]),
        }

    def create_tenant_owner(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        user_id: str,
        display_name: str,
        role: str,
        token_id: str,
        token_hash: str,
        token_prefix: str,
    ) -> None:
        tenant_id = safe_id(tenant_id, "tenant_id")
        user_id = safe_id(user_id, "user_id")
        created_at = utc_now()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO tenants VALUES (?, ?, 'active', ?)",
                (tenant_id, tenant_name.strip(), created_at),
            )
            connection.execute(
                """
                INSERT INTO users VALUES (?, ?, 'active', ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (user_id, display_name.strip(), created_at),
            )
            connection.execute(
                """
                INSERT INTO memberships
                    (tenant_id, user_id, role, status, created_at)
                VALUES (?, ?, ?, 'active', ?)
                """,
                (tenant_id, user_id, role, created_at),
            )
            connection.execute(
                """
                INSERT INTO api_tokens
                    (token_id, tenant_id, user_id, token_hash, token_prefix,
                     status, created_at, expires_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, NULL, NULL)
                """,
                (
                    safe_id(token_id, "token_id"),
                    tenant_id,
                    user_id,
                    token_hash,
                    token_prefix,
                    created_at,
                ),
            )

    def add_membership(
        self,
        *,
        tenant_id: str,
        user_id: str,
        display_name: str,
        role: str,
    ) -> dict[str, Any]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        user_id = safe_id(user_id, "user_id")
        created_at = utc_now()
        with self._transaction() as connection:
            tenant = connection.execute(
                "SELECT status FROM tenants WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            if tenant is None or tenant["status"] != "active":
                raise ValueError("tenant is not active")
            connection.execute(
                """
                INSERT INTO users VALUES (?, ?, 'active', ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (user_id, display_name.strip(), created_at),
            )
            connection.execute(
                """
                INSERT INTO memberships
                    (tenant_id, user_id, role, status, created_at)
                VALUES (?, ?, ?, 'active', ?)
                """,
                (tenant_id, user_id, role, created_at),
            )
        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "display_name": display_name.strip(),
            "role": role,
            "created_at": created_at,
        }

    def membership(self, tenant_id: str, user_id: str) -> dict[str, Any] | None:
        tenant_id = safe_id(tenant_id, "tenant_id")
        user_id = safe_id(user_id, "user_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT m.tenant_id, m.user_id, m.role,
                       m.status AS membership_status, m.created_at,
                       t.status AS tenant_status, u.status AS user_status,
                       u.display_name
                FROM memberships AS m
                JOIN tenants AS t ON t.tenant_id = m.tenant_id
                JOIN users AS u ON u.user_id = m.user_id
                WHERE m.tenant_id = ? AND m.user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def list_members(self, tenant_id: str) -> list[dict[str, Any]]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.tenant_id, m.user_id, u.display_name, m.role,
                       m.status, m.created_at
                FROM memberships AS m
                JOIN users AS u ON u.user_id = m.user_id
                WHERE m.tenant_id = ?
                ORDER BY m.created_at, m.user_id
                """,
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_owner_count(self, tenant_id: str) -> int:
        tenant_id = safe_id(tenant_id, "tenant_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM memberships
                WHERE tenant_id = ? AND role = 'owner' AND status = 'active'
                """,
                (tenant_id,),
            ).fetchone()
        return int(row["count"])

    def update_membership(
        self,
        *,
        tenant_id: str,
        user_id: str,
        role: str,
        status: str,
    ) -> dict[str, Any]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        user_id = safe_id(user_id, "user_id")
        with self._transaction() as connection:
            if self.backend == "postgresql":
                connection.execute(
                    """
                    SELECT user_id FROM memberships
                    WHERE tenant_id = ?
                    FOR UPDATE
                    """,
                    (tenant_id,),
                ).fetchall()
            current = connection.execute(
                """
                SELECT role, status FROM memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
            if current is None:
                raise ValueError("membership does not exist")
            removing_active_owner = (
                current["role"] == "owner"
                and current["status"] == "active"
                and (role != "owner" or status != "active")
            )
            if removing_active_owner:
                owner_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM memberships
                    WHERE tenant_id = ?
                      AND role = 'owner'
                      AND status = 'active'
                    """,
                    (tenant_id,),
                ).fetchone()
                if int(owner_count["count"]) <= 1:
                    raise ValueError(LAST_OWNER_ERROR)
            connection.execute(
                """
                UPDATE memberships SET role = ?, status = ?
                WHERE tenant_id = ? AND user_id = ?
                """,
                (role, status, tenant_id, user_id),
            )
            row = connection.execute(
                """
                SELECT m.tenant_id, m.user_id, u.display_name, m.role,
                       m.status, m.created_at
                FROM memberships AS m
                JOIN users AS u ON u.user_id = m.user_id
                WHERE m.tenant_id = ? AND m.user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
        return dict(row)

    def insert_token(
        self,
        *,
        token_id: str,
        tenant_id: str,
        user_id: str,
        token_hash: str,
        token_prefix: str,
        expires_at: str | None,
    ) -> dict[str, Any]:
        token_id = safe_id(token_id, "token_id")
        tenant_id = safe_id(tenant_id, "tenant_id")
        user_id = safe_id(user_id, "user_id")
        created_at = utc_now()
        with self._transaction() as connection:
            membership_query = """
                SELECT m.status AS membership_status,
                       t.status AS tenant_status,
                       u.status AS user_status
                FROM memberships AS m
                JOIN tenants AS t ON t.tenant_id = m.tenant_id
                JOIN users AS u ON u.user_id = m.user_id
                WHERE m.tenant_id = ? AND m.user_id = ?
                """
            if self.backend == "postgresql":
                membership_query += " FOR UPDATE"
            membership = connection.execute(
                membership_query,
                (tenant_id, user_id),
            ).fetchone()
            if membership is None:
                raise ValueError("user is not a tenant member")
            if any(
                membership[field] != "active"
                for field in (
                    "membership_status",
                    "tenant_status",
                    "user_status",
                )
            ):
                raise ValueError("user membership is not active")
            connection.execute(
                """
                INSERT INTO api_tokens
                    (token_id, tenant_id, user_id, token_hash, token_prefix,
                     status, created_at, expires_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                """,
                (
                    token_id,
                    tenant_id,
                    user_id,
                    token_hash,
                    token_prefix,
                    created_at,
                    expires_at,
                ),
            )
        return {
            "token_id": token_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "token_prefix": token_prefix,
            "status": "active",
            "created_at": created_at,
            "expires_at": expires_at,
        }

    def identity_for_hash(self, token_hash: str) -> dict[str, Any] | None:
        now = utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT tok.token_id, tok.tenant_id, tok.user_id, tok.status,
                       tok.expires_at, m.role, t.status AS tenant_status,
                       u.status AS user_status,
                       m.status AS membership_status
                FROM api_tokens AS tok
                JOIN memberships AS m
                  ON m.tenant_id = tok.tenant_id AND m.user_id = tok.user_id
                JOIN tenants AS t ON t.tenant_id = tok.tenant_id
                JOIN users AS u ON u.user_id = tok.user_id
                WHERE tok.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE api_tokens SET last_used_at = ? WHERE token_id = ?",
                    (now, row["token_id"]),
                )
        return dict(row) if row else None

    def list_tokens(self, tenant_id: str) -> list[dict[str, Any]]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT token_id, tenant_id, user_id, token_prefix, status,
                       created_at, expires_at, last_used_at
                FROM api_tokens
                WHERE tenant_id = ?
                ORDER BY created_at DESC, token_id
                """,
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def revoke_token(self, *, tenant_id: str, token_id: str) -> dict[str, Any]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        token_id = safe_id(token_id, "token_id")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT token_id, tenant_id, user_id, status
                FROM api_tokens
                WHERE tenant_id = ? AND token_id = ?
                """,
                (tenant_id, token_id),
            ).fetchone()
            if row is None:
                raise ValueError("token does not exist")
            if row["status"] == "revoked":
                raise ValueError("token is already revoked")
            connection.execute(
                "UPDATE api_tokens SET status = 'revoked' WHERE token_id = ?",
                (token_id,),
            )
        return {
            "token_id": token_id,
            "tenant_id": tenant_id,
            "user_id": str(row["user_id"]),
            "status": "revoked",
        }

    def create_knowledge_space(
        self,
        *,
        tenant_id: str,
        space_id: str,
        name: str,
        created_by: str,
    ) -> dict[str, Any]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        space_id = safe_id(space_id, "knowledge_space_id")
        created_by = safe_id(created_by, "user_id")
        created_at = utc_now()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_spaces
                    (tenant_id, space_id, name, status, created_by, created_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (tenant_id, space_id, name.strip(), created_by, created_at),
            )
        return {
            "tenant_id": tenant_id,
            "knowledge_space_id": space_id,
            "name": name.strip(),
            "status": "active",
            "created_by": created_by,
            "created_at": created_at,
        }

    def knowledge_space(self, tenant_id: str, space_id: str) -> dict[str, Any] | None:
        tenant_id = safe_id(tenant_id, "tenant_id")
        space_id = safe_id(space_id, "knowledge_space_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT tenant_id, space_id AS knowledge_space_id, name, status,
                       created_by, created_at
                FROM knowledge_spaces
                WHERE tenant_id = ? AND space_id = ?
                """,
                (tenant_id, space_id),
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_spaces(self, tenant_id: str) -> list[dict[str, Any]]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tenant_id, space_id AS knowledge_space_id, name, status,
                       created_by, created_at
                FROM knowledge_spaces
                WHERE tenant_id = ?
                ORDER BY space_id
                """,
                (tenant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_audit(
        self,
        *,
        tenant_id: str,
        actor_user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        trace_id: str,
        outcome: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = (
            utc_now(),
            safe_id(tenant_id, "tenant_id"),
            safe_id(actor_user_id, "user_id"),
            action,
            resource_type,
            resource_id,
            trace_id,
            outcome,
            json.dumps(details or {}, ensure_ascii=False, separators=(",", ":")),
        )
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO audit_events
                    (timestamp, tenant_id, actor_user_id, action, resource_type,
                     resource_id, trace_id, outcome, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record,
            )

    def list_audit(self, tenant_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        tenant_id = safe_id(tenant_id, "tenant_id")
        bounded_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, timestamp, tenant_id, actor_user_id, action,
                       resource_type, resource_id, trace_id, outcome, details_json
                FROM audit_events
                WHERE tenant_id = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (tenant_id, bounded_limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result
