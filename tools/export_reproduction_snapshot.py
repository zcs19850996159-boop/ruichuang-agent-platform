#!/usr/bin/env python3
"""Export a credential-free reproduction snapshot from the live deployment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


PHASE3_ROOT = Path("/root/autodl-tmp/customer_agent_phase3")
APP_ROOT = PHASE3_ROOT / "app"
EXPORT_ROOT = PHASE3_ROOT / "backups" / "github-reproduction"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)


def export_database(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    database_url = urlparse(os.environ["CONTROL_PLANE_DATABASE_URL"])
    database_name = database_url.path.lstrip("/")
    if not database_url.username or not database_name:
        raise RuntimeError("CONTROL_PLANE_DATABASE_URL is incomplete")
    environment = os.environ.copy()
    if database_url.password:
        environment["PGPASSWORD"] = unquote(database_url.password)
    with target.open("wb") as stream:
        subprocess.run(
            [
                "runuser",
                "-u",
                "postgres",
                "--",
                "/usr/lib/postgresql/14/bin/pg_dump",
                "-h",
                database_url.hostname or "127.0.0.1",
                "-p",
                str(database_url.port or 5432),
                "-U",
                unquote(database_url.username),
                "-d",
                database_name,
                "--format=custom",
                "--no-owner",
                "--no-privileges",
            ],
            check=True,
            stdout=stream,
            env=environment,
        )


def export_object_versions(target: Path) -> dict[str, object]:
    import boto3

    endpoint = os.environ["KNOWLEDGE_OBJECT_STORE_ENDPOINT"]
    bucket = os.environ["KNOWLEDGE_OBJECT_STORE_BUCKET"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("KNOWLEDGE_OBJECT_STORE_REGION") or None,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    entries: list[dict[str, object]] = []
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket):
        for item in page.get("Versions", []):
            key = item["Key"]
            version_id = item["VersionId"]
            encoded_key = quote(key, safe="")
            destination = target / "versions" / encoded_key / version_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            response = client.get_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
            )
            with destination.open("wb") as stream:
                shutil.copyfileobj(response["Body"], stream)
            entries.append(
                {
                    "key": key,
                    "version_id": version_id,
                    "is_latest": bool(item.get("IsLatest")),
                    "last_modified": item["LastModified"].isoformat(),
                    "etag": str(item.get("ETag", "")).strip('"'),
                    "size": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "path": destination.relative_to(target).as_posix(),
                }
            )
        for item in page.get("DeleteMarkers", []):
            entries.append(
                {
                    "key": item["Key"],
                    "version_id": item["VersionId"],
                    "is_latest": bool(item.get("IsLatest")),
                    "last_modified": item["LastModified"].isoformat(),
                    "delete_marker": True,
                }
            )
    manifest = {
        "schema_version": "1.0",
        "bucket": bucket,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "bucket": bucket,
        "versions": sum(1 for item in entries if not item.get("delete_marker")),
        "delete_markers": sum(1 for item in entries if item.get("delete_marker")),
    }


def model_inventory() -> list[dict[str, object]]:
    models_root = Path("/root/autodl-tmp/models")
    inventory: list[dict[str, object]] = []
    if not models_root.exists():
        return inventory
    for path in sorted(models_root.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": path.relative_to(models_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return inventory


def main() -> int:
    snapshot = EXPORT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot.mkdir(parents=True)
    export_database(snapshot / "postgres" / "ruichuang_phase3c.dump")
    copy_tree(APP_ROOT / "knowledge_store", snapshot / "knowledge_store")
    copy_tree(APP_ROOT / "outputs" / "rag_assets", snapshot / "rag_assets")
    object_summary = export_object_versions(snapshot / "object_store")
    inventory = {
        "schema_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "application_git_head": subprocess.run(
            ["git", "-C", str(APP_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "object_store": object_summary,
        "models": model_inventory(),
    }
    (snapshot / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256(path)}  {path.relative_to(snapshot).as_posix()}")
    (snapshot / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    archive = snapshot.with_suffix(".tar.gz")
    run("tar", "-C", str(snapshot.parent), "-czf", str(archive), snapshot.name)
    print(json.dumps({"snapshot": str(snapshot), "archive": str(archive)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
