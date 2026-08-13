from __future__ import annotations

import os
from pathlib import Path

from .postgres_store import PostgreSQLControlPlaneStore
from .store import SQLiteControlPlaneStore
from .store_contract import ControlPlaneStore


def create_control_plane_store(
    *,
    default_sqlite_path: str | Path,
) -> ControlPlaneStore:
    """Build the configured store without logging credentials or filesystem paths."""

    database_url = os.environ.get("CONTROL_PLANE_DATABASE_URL", "").strip()
    require_postgres = (
        os.environ.get("CONTROL_PLANE_REQUIRE_POSTGRES", "0").strip() == "1"
    )
    if database_url.startswith(("postgresql://", "postgres://")):
        return PostgreSQLControlPlaneStore(database_url)
    if database_url.startswith("sqlite:///"):
        if require_postgres:
            raise RuntimeError("production control plane requires PostgreSQL")
        return SQLiteControlPlaneStore(database_url.removeprefix("sqlite:///"))
    if database_url:
        raise ValueError("unsupported CONTROL_PLANE_DATABASE_URL scheme")
    if require_postgres:
        raise RuntimeError(
            "CONTROL_PLANE_DATABASE_URL must select PostgreSQL in production"
        )
    legacy_path = os.environ.get("CONTROL_PLANE_DB", "").strip()
    return SQLiteControlPlaneStore(legacy_path or default_sqlite_path)
