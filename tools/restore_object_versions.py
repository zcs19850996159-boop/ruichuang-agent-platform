#!/usr/bin/env python3
"""Restore exported knowledge object versions into a new S3-compatible bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import boto3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.snapshot / "object_store" / "manifest.json").read_text())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["KNOWLEDGE_OBJECT_STORE_ENDPOINT"],
        region_name=os.environ.get("KNOWLEDGE_OBJECT_STORE_REGION") or None,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    try:
        client.head_bucket(Bucket=args.bucket)
    except Exception:
        client.create_bucket(Bucket=args.bucket)
    client.put_bucket_versioning(
        Bucket=args.bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    restored = 0
    for item in sorted(
        (entry for entry in manifest["entries"] if not entry.get("delete_marker")),
        key=lambda entry: (entry["key"], entry["last_modified"]),
    ):
        source = args.snapshot / "object_store" / item["path"]
        if digest(source) != item["sha256"]:
            raise RuntimeError(f"checksum mismatch: {source}")
        with source.open("rb") as stream:
            client.put_object(Bucket=args.bucket, Key=item["key"], Body=stream)
        restored += 1
    print(json.dumps({"bucket": args.bucket, "restored_versions": restored}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
