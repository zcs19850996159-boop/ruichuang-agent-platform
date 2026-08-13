from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable


SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./=-]{0,1023}$")


def safe_object_key(value: str) -> str:
    key = str(value or "").strip().strip("/")
    if (
        not SAFE_KEY.fullmatch(key)
        or "\\" in key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError("invalid object key")
    return key


@runtime_checkable
class ObjectStorage(Protocol):
    backend: str

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        overwrite: bool = False,
    ) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list_keys(self, prefix: str) -> list[str]: ...

    def health(self) -> dict[str, str]: ...


class S3ObjectStorage:
    """S3-compatible object store with an application-owned key prefix."""

    backend = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        server_side_encryption: str | None = None,
    ) -> None:
        if not str(bucket or "").strip():
            raise ValueError("KNOWLEDGE_OBJECT_STORE_BUCKET is required")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 object storage requires boto3") from exc
        self.bucket = str(bucket).strip()
        self.prefix = safe_object_key(prefix)
        self.encryption = str(server_side_encryption or "").strip()
        self.client = boto3.client(
            "s3",
            endpoint_url=(str(endpoint_url).strip() or None) if endpoint_url else None,
            region_name=(str(region_name).strip() or None) if region_name else None,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{safe_object_key(key)}"

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        overwrite: bool = False,
    ) -> None:
        normalized = safe_object_key(key)
        data = bytes(payload)
        if not overwrite and self.exists(normalized):
            if self.get_bytes(normalized) == data:
                return
            raise ValueError("immutable object already exists")
        arguments = {
            "Bucket": self.bucket,
            "Key": self._key(normalized),
            "Body": data,
            "ContentType": content_type,
        }
        if self.encryption:
            arguments["ServerSideEncryption"] = self.encryption
        self.client.put_object(**arguments)

    def get_bytes(self, key: str) -> bytes:
        result = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return result["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except self.client.exceptions.ClientError as exc:
            status = int(
                exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            )
            if status == 404:
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        normalized = safe_object_key(prefix)
        physical_prefix = self._key(normalized).rstrip("/") + "/"
        result: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=physical_prefix,
        ):
            for item in page.get("Contents") or []:
                physical = str(item["Key"])
                result.append(physical.removeprefix(f"{self.prefix}/"))
        return sorted(result)

    def health(self) -> dict[str, str]:
        self.client.head_bucket(Bucket=self.bucket)
        return {"status": "ready", "backend": self.backend}


def create_object_storage_from_environment() -> ObjectStorage | None:
    backend = os.environ.get(
        "KNOWLEDGE_OBJECT_STORE_BACKEND",
        "local",
    ).strip().lower()
    if backend in {"", "local", "filesystem"}:
        return None
    if backend != "s3":
        raise ValueError("unsupported KNOWLEDGE_OBJECT_STORE_BACKEND")
    return S3ObjectStorage(
        bucket=os.environ.get("KNOWLEDGE_OBJECT_STORE_BUCKET", ""),
        prefix=os.environ.get(
            "KNOWLEDGE_OBJECT_STORE_PREFIX",
            "ruichuang/knowledge",
        ),
        endpoint_url=os.environ.get("KNOWLEDGE_OBJECT_STORE_ENDPOINT"),
        region_name=os.environ.get("KNOWLEDGE_OBJECT_STORE_REGION"),
        server_side_encryption=os.environ.get("KNOWLEDGE_OBJECT_STORE_SSE"),
    )
