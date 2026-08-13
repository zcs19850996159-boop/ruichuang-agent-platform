from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CONFIRM_RESTORE = "RESTORE_RUICHUANG_DATA"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_integrity(path: Path) -> None:
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        raise ValueError("SQLite integrity check failed")


def create_sqlite_snapshot(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError("source SQLite database does not exist")
    if source_path == target:
        raise ValueError("snapshot destination must differ from source")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        source_uri = f"file:{source_path}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source_connection:
            with sqlite3.connect(temporary) as target_connection:
                source_connection.backup(target_connection)
        _sqlite_integrity(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "created_at": utc_now(),
        "kind": "sqlite",
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def restore_sqlite_snapshot(
    snapshot: str | Path,
    target: str | Path,
    *,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRM_RESTORE:
        raise ValueError("explicit restore confirmation is required")
    source = Path(snapshot).expanduser().resolve()
    destination = Path(target).expanduser().resolve()
    if not source.is_file():
        raise ValueError("SQLite snapshot does not exist")
    if source == destination:
        raise ValueError("snapshot and restore target must differ")
    _sqlite_integrity(source)
    recovery: Path | None = None
    if destination.exists():
        recovery = destination.parent / (
            f"{destination.name}.pre-restore-{uuid.uuid4().hex[:12]}"
        )
        create_sqlite_snapshot(destination, recovery)
    restored = create_sqlite_snapshot(source, destination)
    return {
        **restored,
        "restored_at": utc_now(),
        "recovery_snapshot": str(recovery) if recovery else None,
    }


def create_knowledge_archive(
    knowledge_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    source = Path(knowledge_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("knowledge root does not exist")
    if source == target or source in target.parents:
        raise ValueError("archive destination cannot be inside knowledge root")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    included: list[str] = []
    try:
        with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for name in ("versions", "active", "audit"):
                path = source / name
                if path.exists():
                    archive.add(path, arcname=name, recursive=True)
                    included.append(name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "created_at": utc_now(),
        "kind": "knowledge",
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "included": included,
    }


def restore_knowledge_archive(
    archive_path: str | Path,
    target_root: str | Path,
    *,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRM_RESTORE:
        raise ValueError("explicit restore confirmation is required")
    source = Path(archive_path).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    if not source.is_file():
        raise ValueError("knowledge archive does not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.restore-",
            dir=target.parent,
        )
    )
    recovery: Path | None = None
    try:
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if (
                    member.issym()
                    or member.islnk()
                    or not (member.isfile() or member.isdir())
                    or not parts
                    or parts[0] not in {"versions", "active", "audit"}
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    raise ValueError("unsafe knowledge archive member")
            archive.extractall(temporary, members=members, filter="data")
        if target.exists():
            recovery = target.parent / (
                f"{target.name}.pre-restore-{uuid.uuid4().hex[:12]}"
            )
            os.replace(target, recovery)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "restored_at": utc_now(),
        "kind": "knowledge",
        "archive_sha256": sha256_file(source),
        "recovery_directory": str(recovery) if recovery else None,
    }


def write_manifest(
    destination: str | Path,
    *,
    artifacts: list[dict[str, Any]],
    application_version: str,
) -> None:
    payload = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "application_version": application_version,
        "artifacts": artifacts,
    }
    Path(destination).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
