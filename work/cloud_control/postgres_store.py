from __future__ import annotations

import os
import threading
from typing import Any

from .migrations import apply_postgres_migrations
from .store import SQLiteControlPlaneStore


def _postgres_sql(query: str) -> str:
    if query.strip().upper() == "BEGIN IMMEDIATE":
        return "BEGIN"
    return query.replace("?", "%s")


class _PooledConnectionProxy:
    """Expose the small DB-API surface used by the shared SQL store methods."""

    def __init__(self, pool: Any) -> None:
        self._context = pool.connection()
        self._connection = self._context.__enter__()
        self._closed = False

    def execute(self, query: str, parameters: Any = ()) -> Any:
        return self._connection.execute(_postgres_sql(query), parameters)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._context.__exit__(None, None, None)

    def __enter__(self) -> "_PooledConnectionProxy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.__exit__(exc_type, exc, traceback)


class PostgreSQLControlPlaneStore(SQLiteControlPlaneStore):
    """PostgreSQL implementation with a bounded synchronous connection pool."""

    backend = "postgresql"

    def __init__(
        self,
        dsn: str,
        *,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        normalized = str(dsn or "").strip()
        if not normalized.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgreSQL control-plane DSN is required")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL backend requires psycopg and psycopg-pool"
            ) from exc

        self.dsn = normalized
        self._lock = threading.RLock()
        minimum = max(
            1,
            int(
                pool_min_size
                if pool_min_size is not None
                else os.environ.get("CONTROL_PLANE_DB_POOL_MIN", "1")
            ),
        )
        maximum = max(
            minimum,
            int(
                pool_max_size
                if pool_max_size is not None
                else os.environ.get("CONTROL_PLANE_DB_POOL_MAX", "8")
            ),
        )
        timeout = max(
            1.0,
            float(
                connect_timeout
                if connect_timeout is not None
                else os.environ.get("CONTROL_PLANE_DB_CONNECT_TIMEOUT", "10")
            ),
        )
        self._pool = ConnectionPool(
            conninfo=self.dsn,
            min_size=minimum,
            max_size=maximum,
            timeout=timeout,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        self._pool.wait(timeout=timeout)
        self._initialize()

    def _connect(self) -> _PooledConnectionProxy:
        return _PooledConnectionProxy(self._pool)

    def _initialize(self) -> None:
        with self._pool.connection() as connection:
            self.schema_version = apply_postgres_migrations(connection)

    def close(self) -> None:
        self._pool.close()
