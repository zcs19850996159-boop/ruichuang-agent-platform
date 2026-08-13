#!/usr/bin/env python3
"""Enable safe MinIO lifecycle and SSE-S3, then encrypt current object versions."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config


INFRA_ROOT = Path("/root/autodl-tmp/customer_agent_phase3/infra")


def load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key.strip()] = value


load_env(INFRA_ROOT / "secrets" / "phase3c.env")

bucket = os.environ["KNOWLEDGE_OBJECT_STORE_BUCKET"]
client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("KNOWLEDGE_OBJECT_STORE_ENDPOINT", "http://127.0.0.1:59000"),
    region_name=os.environ.get("KNOWLEDGE_OBJECT_STORE_REGION", "us-east-1"),
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
)

client.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
client.put_bucket_encryption(
    Bucket=bucket,
    ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    },
)
client.put_object_lock_configuration(
    Bucket=bucket,
    ObjectLockConfiguration={
        "ObjectLockEnabled": "Enabled",
        "Rule": {
            "DefaultRetention": {
                "Mode": "GOVERNANCE",
                "Days": 30,
            }
        },
    },
)
client.put_bucket_lifecycle_configuration(
    Bucket=bucket,
    LifecycleConfiguration={
        "Rules": [
            {
                "ID": "phase3c-safe-retention",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 180},
            }
        ]
    },
)

migrated = 0
verified = 0
failures: list[str] = []
paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket):
    for item in page.get("Contents", []):
        key = item["Key"]
        head = client.head_object(Bucket=bucket, Key=key)
        if head.get("ServerSideEncryption") != "AES256":
            client.copy_object(
                Bucket=bucket,
                Key=key,
                CopySource={"Bucket": bucket, "Key": key},
                MetadataDirective="COPY",
                TaggingDirective="COPY",
                ServerSideEncryption="AES256",
            )
            migrated += 1
            head = client.head_object(Bucket=bucket, Key=key)
        if head.get("ServerSideEncryption") == "AES256":
            verified += 1
        else:
            failures.append(key)

if failures:
    raise SystemExit(f"objects without AES256 after migration: {failures!r}")

encryption = client.get_bucket_encryption(Bucket=bucket)
lifecycle = client.get_bucket_lifecycle_configuration(Bucket=bucket)
versioning = client.get_bucket_versioning(Bucket=bucket)
object_lock = client.get_object_lock_configuration(Bucket=bucket)
print(
    {
        "bucket": bucket,
        "migrated_current_objects": migrated,
        "verified_encrypted_current_objects": verified,
        "versioning": versioning.get("Status"),
        "default_encryption": encryption["ServerSideEncryptionConfiguration"]["Rules"],
        "lifecycle_rules": lifecycle.get("Rules", []),
        "object_lock": object_lock.get("ObjectLockConfiguration", {}),
    }
)
