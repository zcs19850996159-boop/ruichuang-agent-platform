from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cloud_control import ControlPlaneError, ControlPlaneService, SQLiteControlPlaneStore


PEPPER = "phase3-test-pepper-with-at-least-32-bytes"


def service(tmp_path: Path) -> ControlPlaneService:
    return ControlPlaneService(
        SQLiteControlPlaneStore(tmp_path / "control.sqlite3"),
        token_pepper=PEPPER,
    )


def bootstrap(
    control: ControlPlaneService,
    tenant_id: str,
    owner_id: str,
) -> tuple[dict, object]:
    result = control.bootstrap_tenant(
        tenant_id=tenant_id,
        tenant_name=f"{tenant_id} company",
        owner_user_id=owner_id,
        owner_display_name=owner_id,
    )
    principal = control.authenticate(result["api_token"])
    assert principal is not None
    return result, principal


def test_tokens_are_returned_once_and_persisted_only_as_hashes(
    tmp_path: Path,
) -> None:
    control = service(tmp_path)
    result, principal = bootstrap(control, "tenant-a", "alice")
    assert principal.tenant_id == "tenant-a"
    assert principal.role == "owner"
    assert result["api_token"].startswith("rcp_")
    assert len(result["api_token"]) >= 47

    with sqlite3.connect(tmp_path / "control.sqlite3") as connection:
        row = connection.execute(
            "SELECT token_hash, token_prefix FROM api_tokens"
        ).fetchone()
    assert row is not None
    assert row[0] != result["api_token"]
    assert len(row[0]) == 64
    assert row[1] == result["api_token"][:12]
    assert result["api_token"] not in (tmp_path / "control.sqlite3").read_bytes().decode(
        "latin-1"
    )


def test_roles_and_cross_tenant_isolation(tmp_path: Path) -> None:
    control = service(tmp_path)
    _, owner_a = bootstrap(control, "tenant-a", "alice")
    _, owner_b = bootstrap(control, "tenant-b", "bob")

    control.create_knowledge_space(
        owner_a,
        tenant_id="tenant-a",
        space_id="manuals",
        name="Manuals",
    )
    control.create_member(
        owner_a,
        tenant_id="tenant-a",
        user_id="agent-a",
        display_name="Agent A",
        role="agent",
    )
    issued = control.issue_token(
        owner_a,
        tenant_id="tenant-a",
        user_id="agent-a",
    )
    agent_a = control.authenticate(issued["api_token"])
    assert agent_a is not None
    assert agent_a.can("customer:answer")
    assert not agent_a.can("knowledge:write")
    assert control.require_space(agent_a, "manuals", "customer:answer")

    with pytest.raises(ControlPlaneError) as denied_create:
        control.create_knowledge_space(
            agent_a,
            tenant_id="tenant-a",
            space_id="private",
            name="Private",
        )
    assert denied_create.value.http_status == 403

    with pytest.raises(ControlPlaneError) as denied_cross_tenant:
        control.list_knowledge_spaces(owner_a, tenant_id="tenant-b")
    assert denied_cross_tenant.value.http_status == 403

    with pytest.raises(ControlPlaneError) as reverse_denied:
        control.list_knowledge_spaces(owner_b, tenant_id="tenant-a")
    assert reverse_denied.value.http_status == 403


def test_audit_queries_are_tenant_scoped(tmp_path: Path) -> None:
    control = service(tmp_path)
    _, owner_a = bootstrap(control, "tenant-a", "alice")
    _, owner_b = bootstrap(control, "tenant-b", "bob")
    control.create_knowledge_space(
        owner_a,
        tenant_id="tenant-a",
        space_id="manuals-a",
        name="A",
    )
    control.create_knowledge_space(
        owner_b,
        tenant_id="tenant-b",
        space_id="manuals-b",
        name="B",
    )

    audit_a = control.list_audit(owner_a, tenant_id="tenant-a")
    assert audit_a
    assert {event["tenant_id"] for event in audit_a} == {"tenant-a"}
    assert any(event["action"] == "knowledge_space.create" for event in audit_a)
    assert all("api_token" not in event["details"] for event in audit_a)


def test_unknown_and_malformed_tokens_do_not_authenticate(tmp_path: Path) -> None:
    control = service(tmp_path)
    bootstrap(control, "tenant-a", "alice")
    assert control.authenticate("not-a-control-token") is None
    assert control.authenticate("rcp_unknown") is None
    assert control.authenticate_authorization("Basic abc") is None


def test_revoked_and_expired_tokens_do_not_authenticate(tmp_path: Path) -> None:
    control = service(tmp_path)
    _, owner = bootstrap(control, "tenant-a", "alice")
    control.create_member(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        display_name="Agent A",
        role="agent",
    )
    issued = control.issue_token(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
    )
    assert control.authenticate(issued["api_token"]) is not None
    result = control.revoke_token(
        owner,
        tenant_id="tenant-a",
        token_id=issued["token_id"],
    )
    assert result["status"] == "revoked"
    assert control.authenticate(issued["api_token"]) is None

    expiring = control.issue_token(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    with sqlite3.connect(tmp_path / "control.sqlite3") as connection:
        connection.execute(
            "UPDATE api_tokens SET expires_at = ? WHERE token_id = ?",
            (
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                expiring["token_id"],
            ),
        )
        connection.commit()
    assert control.authenticate(expiring["api_token"]) is None


def test_pepper_must_be_long_enough(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 32"):
        ControlPlaneService(
            SQLiteControlPlaneStore(tmp_path / "control.sqlite3"),
            token_pepper="short",
        )
