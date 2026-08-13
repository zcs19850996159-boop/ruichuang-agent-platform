from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_lifecycle import KnowledgeLifecycleService, KnowledgeVersionStore
from knowledge_lifecycle.object_storage import ObjectStorage, safe_object_key


class MemoryObjectStorage:
    backend = "memory"

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put_bytes(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        overwrite: bool = False,
    ) -> None:
        del content_type
        normalized = safe_object_key(key)
        data = bytes(payload)
        if not overwrite and normalized in self.values:
            if self.values[normalized] == data:
                return
            raise ValueError("immutable object already exists")
        self.values[normalized] = data

    def get_bytes(self, key: str) -> bytes:
        return self.values[safe_object_key(key)]

    def exists(self, key: str) -> bool:
        return safe_object_key(key) in self.values

    def list_keys(self, prefix: str) -> list[str]:
        normalized = safe_object_key(prefix).rstrip("/") + "/"
        return sorted(key for key in self.values if key.startswith(normalized))

    def health(self) -> dict[str, str]:
        return {"status": "ready", "backend": self.backend}


def publish_text_manual(
    service: KnowledgeLifecycleService,
    manual: Path,
    *,
    tenant: str,
    version: str,
) -> None:
    staged = service.ingest(
        str(manual),
        product_id="x300",
        actor="alice",
        tenant_id=tenant,
        space_id="manuals",
    )
    assert service.evaluate(staged["staging_id"])["publishable"] is True
    service.publish(
        staged["staging_id"],
        tenant_id=tenant,
        space_id="manuals",
        version=version,
        approved_by="alice",
    )


def test_object_store_publish_is_tenant_scoped_and_cache_can_rehydrate(
    tmp_path: Path,
) -> None:
    objects = MemoryObjectStorage()
    assert isinstance(objects, ObjectStorage)
    source = tmp_path / "manual.txt"
    source.write_text("X300 开机方法\n按住电源键三秒启动设备。", encoding="utf-8")
    lifecycle = KnowledgeLifecycleService(
        KnowledgeVersionStore(
            tmp_path / "writer-cache",
            object_storage=objects,
        )
    )
    publish_text_manual(
        lifecycle,
        source,
        tenant="tenant-a",
        version="knowledge-v1",
    )

    keys = sorted(objects.values)
    assert any(
        key.startswith("tenants/tenant-a/spaces/manuals/versions/knowledge-v1/")
        for key in keys
    )
    assert not any("tenant-b" in key for key in keys)

    reader = KnowledgeLifecycleService(
        KnowledgeVersionStore(
            tmp_path / "reader-cache",
            object_storage=objects,
        )
    )
    assert reader.active_version(
        tenant_id="tenant-a",
        space_id="manuals",
    )["version"] == "knowledge-v1"
    assert reader.list_versions(
        tenant_id="tenant-a",
        space_id="manuals",
    )[0]["active"] is True
    result = reader.search(
        "怎样启动 X300？",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    assert result["hits"]
    assert (
        tmp_path
        / "reader-cache"
        / "versions"
        / "tenant-a--manuals"
        / "knowledge-v1"
        / "manifest.json"
    ).is_file()
    assert reader.list_versions(
        tenant_id="tenant-b",
        space_id="manuals",
    ) == []


def test_object_store_version_is_immutable(tmp_path: Path) -> None:
    objects = MemoryObjectStorage()
    source = tmp_path / "manual.txt"
    source.write_text("按下电源键开机。", encoding="utf-8")
    lifecycle = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "cache", object_storage=objects)
    )
    publish_text_manual(
        lifecycle,
        source,
        tenant="tenant-a",
        version="knowledge-v1",
    )
    with pytest.raises(ValueError, match="immutable"):
        publish_text_manual(
            lifecycle,
            source,
            tenant="tenant-a",
            version="knowledge-v1",
        )


@pytest.mark.parametrize(
    "value",
    ["../secret", "tenant/../../secret", "/absolute/../secret", "a\\b"],
)
def test_unsafe_object_keys_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="invalid object key"):
        safe_object_key(value)
