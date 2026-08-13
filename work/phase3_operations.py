from __future__ import annotations

import argparse
import json

from operations import (
    CONFIRM_RESTORE,
    create_knowledge_archive,
    create_sqlite_snapshot,
    restore_knowledge_archive,
    restore_sqlite_snapshot,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Phase 3 local acceptance backup and recovery commands",
    )
    commands = result.add_subparsers(dest="command", required=True)
    sqlite_backup = commands.add_parser("sqlite-backup")
    sqlite_backup.add_argument("--source", required=True)
    sqlite_backup.add_argument("--output", required=True)
    sqlite_restore = commands.add_parser("sqlite-restore")
    sqlite_restore.add_argument("--snapshot", required=True)
    sqlite_restore.add_argument("--target", required=True)
    sqlite_restore.add_argument("--confirm", required=True)
    knowledge_backup = commands.add_parser("knowledge-backup")
    knowledge_backup.add_argument("--source", required=True)
    knowledge_backup.add_argument("--output", required=True)
    knowledge_restore = commands.add_parser("knowledge-restore")
    knowledge_restore.add_argument("--archive", required=True)
    knowledge_restore.add_argument("--target", required=True)
    knowledge_restore.add_argument("--confirm", required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "sqlite-backup":
        result = create_sqlite_snapshot(arguments.source, arguments.output)
    elif arguments.command == "sqlite-restore":
        result = restore_sqlite_snapshot(
            arguments.snapshot,
            arguments.target,
            confirmation=arguments.confirm,
        )
    elif arguments.command == "knowledge-backup":
        result = create_knowledge_archive(arguments.source, arguments.output)
    else:
        result = restore_knowledge_archive(
            arguments.archive,
            arguments.target,
            confirmation=arguments.confirm,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
