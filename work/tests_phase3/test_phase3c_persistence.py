from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cloud_control import (
    ControlPlaneStore,
    SQLiteControlPlaneStore,
    create_control_plane_store,
)
from cloud_control.migrations import LATEST_SCHEMA_VERSION
from cloud_control.postgres_store import _postgres_sql


def test_fresh_sqlite_database_reaches_latest_schema(tmp_path: Path) -> None:
    store = SQLiteControlPlaneStore(tmp_path / "control.sqlite3")
    assert isinstance(store, ControlPlaneStore)
    assert store.health() == {
        "status": "ready",
        "backend": "sqlite",
        "schema_version": LATEST_SCHEMA_VERSION,
    }
    with sqlite3.connect(tmp_path / "control.sqlite3") as connection:
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(api_tokens)").fetchall()
        }
    assert "idx_tokens_hash" in indexes
    assert "idx_tokens_tenant_status" in indexes


def test_schema_v2_migrates_to_v3_without_losing_tenant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.sqlite3"
    store = SQLiteControlPlaneStore(path)
    control_health = store.health()
    assert control_health["schema_version"] == 3
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO tenants VALUES ('tenant-a', 'Tenant A', 'active', 'now')"
        )
        connection.execute(
            "UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'"
        )
        connection.commit()
    migrated = SQLiteControlPlaneStore(path)
    assert migrated.health()["schema_version"] == 3
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT name FROM tenants WHERE tenant_id = 'tenant-a'"
        ).fetchone()[0] == "Tenant A"


def test_newer_database_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "control.sqlite3"
    SQLiteControlPlaneStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = '999' WHERE key = 'schema_version'"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="newer than application"):
        SQLiteControlPlaneStore(path)


def test_store_factory_supports_sqlite_url_and_production_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "factory.sqlite3"
    monkeypatch.setenv("CONTROL_PLANE_DATABASE_URL", f"sqlite:///{target}")
    monkeypatch.delenv("CONTROL_PLANE_REQUIRE_POSTGRES", raising=False)
    store = create_control_plane_store(default_sqlite_path=tmp_path / "unused.sqlite3")
    assert store.health()["backend"] == "sqlite"

    monkeypatch.setenv("CONTROL_PLANE_REQUIRE_POSTGRES", "1")
    with pytest.raises(RuntimeError, match="requires PostgreSQL"):
        create_control_plane_store(default_sqlite_path=tmp_path / "unused.sqlite3")


def test_postgres_placeholder_adapter_does_not_change_begin_semantics() -> None:
    assert _postgres_sql("BEGIN IMMEDIATE") == "BEGIN"
    assert _postgres_sql("SELECT * FROM tenants WHERE tenant_id = ?") == (
        "SELECT * FROM tenants WHERE tenant_id = %s"
    )
