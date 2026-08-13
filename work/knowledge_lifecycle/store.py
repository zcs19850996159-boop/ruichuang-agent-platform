from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .object_storage import ObjectStorage, create_object_storage_from_environment


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_id(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_ID.fullmatch(normalized):
        raise ValueError(f"invalid {field}")
    return normalized


class KnowledgeVersionStore:
    """Filesystem store with immutable releases and atomic active pointers."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        object_storage: ObjectStorage | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[2] / "knowledge_store"
        self.root = Path(root or os.environ.get("KNOWLEDGE_STORE_ROOT", default_root)).resolve()
        self.staging = self.root / "staging"
        self.versions = self.root / "versions"
        self.active = self.root / "active"
        self.audit = self.root / "audit"
        for directory in (self.staging, self.versions, self.active, self.audit):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.objects = object_storage

    @classmethod
    def from_environment(cls) -> "KnowledgeVersionStore":
        return cls(object_storage=create_object_storage_from_environment())

    def health(self) -> dict[str, str]:
        if self.objects is None:
            return {"status": "ready", "backend": "filesystem"}
        return self.objects.health()

    def scope_key(self, tenant_id: str, space_id: str) -> str:
        return f"{safe_id(tenant_id, 'tenant_id')}--{safe_id(space_id, 'knowledge_space_id')}"

    def staging_path(self, staging_id: str) -> Path:
        return self.staging / safe_id(staging_id, "staging_id")

    def version_path(self, tenant_id: str, space_id: str, version: str) -> Path:
        target = (
            self.versions
            / self.scope_key(tenant_id, space_id)
            / safe_id(version, "version")
        )
        if self.objects is not None and not target.is_dir():
            self._materialize_version(tenant_id, space_id, version, target)
        return target

    def active_pointer(self, tenant_id: str, space_id: str) -> Path:
        return self.active / f"{self.scope_key(tenant_id, space_id)}.json"

    def object_version_prefix(
        self,
        tenant_id: str,
        space_id: str,
        version: str,
    ) -> str:
        return (
            f"tenants/{safe_id(tenant_id, 'tenant_id')}/spaces/"
            f"{safe_id(space_id, 'knowledge_space_id')}/versions/"
            f"{safe_id(version, 'version')}"
        )

    def object_active_key(self, tenant_id: str, space_id: str) -> str:
        return (
            f"tenants/{safe_id(tenant_id, 'tenant_id')}/spaces/"
            f"{safe_id(space_id, 'knowledge_space_id')}/active.json"
        )

    def read_active(self, tenant_id: str, space_id: str) -> dict[str, Any] | None:
        path = self.active_pointer(tenant_id, space_id)
        if self.objects is not None:
            key = self.object_active_key(tenant_id, space_id)
            if not self.objects.exists(key):
                return None
            payload = self.objects.get_bytes(key)
            path.write_bytes(payload)
            return json.loads(payload.decode("utf-8"))
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_versions(self, tenant_id: str, space_id: str) -> list[dict[str, Any]]:
        scope = self.versions / self.scope_key(tenant_id, space_id)
        active = self.read_active(tenant_id, space_id)
        active_version = str((active or {}).get("version") or "")
        if self.objects is not None:
            base = (
                f"tenants/{safe_id(tenant_id, 'tenant_id')}/spaces/"
                f"{safe_id(space_id, 'knowledge_space_id')}/versions"
            )
            manifest_keys = [
                key
                for key in self.objects.list_keys(base)
                if key.endswith("/manifest.json")
            ]
            versions: list[dict[str, Any]] = []
            for manifest_key in sorted(manifest_keys, reverse=True):
                version = manifest_key.split("/")[-2]
                manifest = json.loads(
                    self.objects.get_bytes(manifest_key).decode("utf-8")
                )
                quality_key = (
                    f"{manifest_key.rsplit('/', 1)[0]}/quality_report.json"
                )
                quality = (
                    json.loads(self.objects.get_bytes(quality_key).decode("utf-8"))
                    if self.objects.exists(quality_key)
                    else {}
                )
                versions.append(
                    self._version_summary(
                        manifest,
                        quality,
                        fallback_version=version,
                        active_version=active_version,
                    )
                )
            return versions
        if not scope.is_dir():
            return []
        versions: list[dict[str, Any]] = []
        for version_path in sorted(
            (path for path in scope.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        ):
            manifest_path = version_path / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            quality_path = version_path / "quality_report.json"
            quality = (
                json.loads(quality_path.read_text(encoding="utf-8"))
                if quality_path.is_file()
                else {}
            )
            versions.append(
                self._version_summary(
                    manifest,
                    quality,
                    fallback_version=version_path.name,
                    active_version=active_version,
                )
            )
        return versions

    @staticmethod
    def _version_summary(
        manifest: dict[str, Any],
        quality: dict[str, Any],
        *,
        fallback_version: str,
        active_version: str,
    ) -> dict[str, Any]:
        document = manifest.get("document") or {}
        version = str(manifest.get("version") or fallback_version)
        return {
            "version": version,
            "active": version == active_version,
            "published_at": manifest.get("published_at"),
            "approved_by": manifest.get("approved_by"),
            "immutable": bool(manifest.get("immutable")),
            "product_id": manifest.get("product_id"),
            "document": {
                "document_id": document.get("document_id"),
                "original_name": document.get("original_name"),
                "sha256": document.get("sha256"),
                "mime": document.get("mime"),
                "pages": document.get("pages"),
            },
            "chunk_count": int(manifest.get("chunks") or 0),
            "image_count": len(manifest.get("images") or []),
            "quality": {
                "publishable": quality.get("publishable"),
                "blockers": quality.get("blockers") or [],
                "metrics": quality.get("metrics") or {},
            },
        }

    def activate(self, tenant_id: str, space_id: str, version: str, actor: str) -> dict[str, Any]:
        target = self.version_path(tenant_id, space_id, version)
        if not target.is_dir():
            raise ValueError("knowledge version does not exist")
        previous = self.read_active(tenant_id, space_id)
        payload = {
            "tenant_id": tenant_id,
            "knowledge_space_id": space_id,
            "version": version,
            "previous_version": (previous or {}).get("version"),
            "activated_at": utc_now(),
            "actor": actor,
        }
        pointer = self.active_pointer(tenant_id, space_id)
        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix=f".{pointer.name}.", dir=pointer.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, pointer)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if self.objects is not None:
            self.objects.put_bytes(
                self.object_active_key(tenant_id, space_id),
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                content_type="application/json",
                overwrite=True,
            )
        self.append_audit("activate", payload)
        return payload

    def publish_staging(
        self,
        staging_id: str,
        *,
        tenant_id: str,
        space_id: str,
        version: str,
        approved_by: str,
    ) -> dict[str, Any]:
        source = self.staging_path(staging_id)
        if not source.is_dir():
            raise ValueError("staging package does not exist")
        report_path = source / "quality_report.json"
        if not report_path.exists():
            raise ValueError("knowledge package has not been evaluated")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("publishable") is not True:
            raise ValueError("knowledge quality gate did not pass")
        target = self.version_path(tenant_id, space_id, version)
        object_manifest = (
            f"{self.object_version_prefix(tenant_id, space_id, version)}"
            "/manifest.json"
        )
        if self.objects is not None and self.objects.exists(object_manifest):
            raise ValueError("published knowledge versions are immutable")
        with self._lock:
            if target.exists():
                raise ValueError("published knowledge versions are immutable")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "tenant_id": tenant_id,
                    "knowledge_space_id": space_id,
                    "version": version,
                    "approved_by": approved_by,
                    "published_at": utc_now(),
                    "immutable": True,
                }
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            # Immutability is enforced by never accepting an existing target.
            # Storage-level WORM/retention policies can additionally protect
            # this directory in enterprise deployments.
            if self.objects is not None:
                self._publish_version_objects(
                    target,
                    tenant_id=tenant_id,
                    space_id=space_id,
                    version=version,
                )
        activation = self.activate(tenant_id, space_id, version, approved_by)
        return {"version": version, "manifest": manifest, "activation": activation}

    def rollback(self, tenant_id: str, space_id: str, target_version: str, actor: str) -> dict[str, Any]:
        current = self.read_active(tenant_id, space_id)
        if current and current.get("version") == target_version:
            raise ValueError("target version is already active")
        result = self.activate(tenant_id, space_id, target_version, actor)
        self.append_audit("rollback", {**result, "from_version": (current or {}).get("version")})
        return result

    def append_audit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "event": event, **payload}
        path = self.audit / f"{datetime.now(UTC).date().isoformat()}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if self.objects is not None:
            tenant_id = safe_id(
                str(payload.get("tenant_id") or "system"),
                "tenant_id",
            )
            key = (
                f"tenants/{tenant_id}/audit/"
                f"{datetime.now(UTC).date().isoformat()}/{uuid.uuid4().hex}.json"
            )
            self.objects.put_bytes(
                key,
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                content_type="application/json",
            )

    def _publish_version_objects(
        self,
        target: Path,
        *,
        tenant_id: str,
        space_id: str,
        version: str,
    ) -> None:
        assert self.objects is not None
        prefix = self.object_version_prefix(tenant_id, space_id, version)
        files = sorted(path for path in target.rglob("*") if path.is_file())
        manifest = target / "manifest.json"
        for path in [item for item in files if item != manifest] + [manifest]:
            relative = path.relative_to(target).as_posix()
            content_type = (
                "application/json"
                if path.suffix.lower() in {".json", ".jsonl"}
                else "application/octet-stream"
            )
            self.objects.put_bytes(
                f"{prefix}/{relative}",
                path.read_bytes(),
                content_type=content_type,
            )

    def _materialize_version(
        self,
        tenant_id: str,
        space_id: str,
        version: str,
        target: Path,
    ) -> None:
        assert self.objects is not None
        prefix = self.object_version_prefix(tenant_id, space_id, version)
        keys = self.objects.list_keys(prefix)
        if not any(key == f"{prefix}/manifest.json" for key in keys):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{safe_id(version, 'version')}.",
                dir=target.parent,
            )
        )
        try:
            for key in keys:
                relative = key.removeprefix(f"{prefix}/")
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.objects.get_bytes(key))
            with self._lock:
                if not target.exists():
                    os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
