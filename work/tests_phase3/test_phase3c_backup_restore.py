from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from operations import (
    CONFIRM_RESTORE,
    create_knowledge_archive,
    create_sqlite_snapshot,
    restore_knowledge_archive,
    restore_sqlite_snapshot,
)


def sqlite_with_value(path: Path, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE values_table(value TEXT NOT NULL)")
        connection.execute("INSERT INTO values_table VALUES (?)", (value,))


def read_value(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT value FROM values_table").fetchone()[0]


def test_sqlite_backup_restore_preserves_recovery_snapshot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.sqlite3"
    sqlite_with_value(database, "before-backup")
    snapshot = tmp_path / "control.snapshot.sqlite3"
    metadata = create_sqlite_snapshot(database, snapshot)
    assert metadata["sha256"]
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE values_table SET value = 'after-backup'")
    restored = restore_sqlite_snapshot(
        snapshot,
        database,
        confirmation=CONFIRM_RESTORE,
    )
    assert read_value(database) == "before-backup"
    recovery = Path(restored["recovery_snapshot"])
    assert recovery.is_file()
    assert read_value(recovery) == "after-backup"


def test_restore_requires_explicit_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "target.sqlite3"
    sqlite_with_value(source, "source")
    with pytest.raises(ValueError, match="confirmation"):
        restore_sqlite_snapshot(source, target, confirmation="yes")


def test_knowledge_archive_excludes_staging_and_restores_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "knowledge"
    (source / "versions" / "tenant-a--manuals" / "v1").mkdir(parents=True)
    (source / "versions" / "tenant-a--manuals" / "v1" / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (source / "active").mkdir()
    (source / "active" / "tenant-a--manuals.json").write_text(
        '{"version":"v1"}',
        encoding="utf-8",
    )
    (source / "audit").mkdir()
    (source / "audit" / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "staging").mkdir()
    (source / "staging" / "secret.tmp").write_text("temporary", encoding="utf-8")
    archive = tmp_path / "knowledge.tar.gz"
    metadata = create_knowledge_archive(source, archive)
    assert metadata["included"] == ["versions", "active", "audit"]

    restored = tmp_path / "restored"
    restore_knowledge_archive(
        archive,
        restored,
        confirmation=CONFIRM_RESTORE,
    )
    assert (restored / "versions" / "tenant-a--manuals" / "v1" / "manifest.json").is_file()
    assert not (restored / "staging").exists()
