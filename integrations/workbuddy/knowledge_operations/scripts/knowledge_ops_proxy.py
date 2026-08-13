#!/usr/bin/env python3
"""Tenant-scoped MCP proxy for Ruichuang enterprise knowledge operations."""

from __future__ import annotations

import hmac
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from typing import Any
from urllib import error, parse, request


PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "ruichuang-knowledge-operations-proxy"
SERVER_VERSION = "1.1.0"
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ProxyError(Exception):
    """A safe error that may be returned to the MCP host."""


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: dict[str, dict[str, Any]] = {
    "inspect_knowledge_space": {
        "description": "Inspect the authenticated enterprise identity, active knowledge version, and immutable versions.",
        "inputSchema": _object_schema({}),
        "annotations": {
            "title": "Inspect knowledge space",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "upload_product_manual": {
        "description": "Upload one authorized product manual into isolated staging. This never changes the active version.",
        "inputSchema": _object_schema(
            {
                "source_path": {"type": "string", "minLength": 1},
                "product_id": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
            },
            ["source_path", "product_id"],
        ),
        "annotations": {
            "title": "Stage a product manual",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    "analyze_product_materials": {
        "description": "Inspect an authorized local product folder before upload. Detect documents, images, duplicate files, likely versions/models/languages, missing official image references, and ambiguous conflicts. No remote state is changed.",
        "inputSchema": _object_schema(
            {"directory_path": {"type": "string", "minLength": 1}},
            ["directory_path"],
        ),
        "annotations": {
            "title": "Analyze product materials",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    "stage_product_materials": {
        "description": "Package all supported documents and images from an analyzed authorized product folder, then upload the package to Ruichuang isolated staging. The Ruichuang platform performs canonical parsing, chunking, image binding, indexing, and RAG preparation.",
        "inputSchema": _object_schema(
            {
                "directory_path": {"type": "string", "minLength": 1},
                "product_id": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
                "product_name": {"type": "string", "maxLength": 256},
                "models": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 128},
                    "maxItems": 100,
                },
                "conflict_resolution": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "User-confirmed resolution for version/model/policy conflicts reported by analyze_product_materials.",
                },
            },
            ["directory_path", "product_id"],
        ),
        "annotations": {
            "title": "Stage product materials",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    "get_staging_status": {
        "description": "Read tenant-scoped non-path metadata for a staged manual.",
        "inputSchema": _object_schema(
            {"staging_id": {"type": "string", "minLength": 1}},
            ["staging_id"],
        ),
        "annotations": {
            "title": "Inspect staged manual",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "check_ingestion_quality": {
        "description": "Run the ingestion quality, security, duplication, OCR-readiness, and image-completeness gate. This is not competition answer evaluation.",
        "inputSchema": _object_schema(
            {"staging_id": {"type": "string", "minLength": 1}},
            ["staging_id"],
        ),
        "annotations": {
            "title": "Run ingestion quality check",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "diagnose_ingestion_blockers": {
        "description": "Diagnose an isolated staging version and return exact missing image IDs, blocker categories, and the next safe action. File changes require a replacement staging package; active knowledge is never changed.",
        "inputSchema": _object_schema(
            {"staging_id": {"type": "string", "minLength": 1}},
            ["staging_id"],
        ),
        "annotations": {
            "title": "Diagnose ingestion blockers",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "run_knowledge_regression": {
        "description": "Run tenant-scoped staging-versus-active retrieval regression and categorize wrong-product, missing-evidence, missing-term, missing-image, and baseline-regression failures. This is not official competition scoring.",
        "inputSchema": _object_schema(
            {
                "staging_id": {"type": "string", "minLength": 1},
                "cases_path": {"type": "string", "minLength": 1},
                "cases": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 500,
                    "items": {"type": "object"},
                },
            },
            ["staging_id"],
        ),
        "annotations": {
            "title": "Run knowledge regression",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "publish_knowledge_version": {
        "description": "Publish an approved staged manual as an immutable version and atomically activate it. Requires explicit user approval and an exact confirmation phrase.",
        "inputSchema": _object_schema(
            {
                "staging_id": {"type": "string", "minLength": 1},
                "version": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
                "confirmation_phrase": {"type": "string", "minLength": 1},
            },
            ["staging_id", "version", "confirmation_phrase"],
        ),
        "annotations": {
            "title": "Publish knowledge version",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    "rollback_knowledge_version": {
        "description": "Atomically switch the active pointer to an existing immutable version. Requires explicit user approval and an exact confirmation phrase.",
        "inputSchema": _object_schema(
            {
                "target_version": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
                "confirmation_phrase": {"type": "string", "minLength": 1},
            },
            ["target_version", "confirmation_phrase"],
        ),
        "annotations": {
            "title": "Roll back knowledge version",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    "list_audit_events": {
        "description": "List recent tenant-scoped enterprise control-plane audit events.",
        "inputSchema": _object_schema(
            {"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}
        ),
        "annotations": {
            "title": "List knowledge audit events",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    "verify_published_knowledge": {
        "description": "Call the fixed enterprise customer-service core once to verify a published product with a representative question.",
        "inputSchema": _object_schema(
            {"question": {"type": "string", "minLength": 1}},
            ["question"],
        ),
        "annotations": {
            "title": "Verify published product knowledge",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProxyError(f"missing required configuration: {name}")
    return value


def _scope() -> tuple[str, str]:
    tenant = _required_env("RUICHUANG_KNOWLEDGE_TENANT_ID")
    space = _required_env("RUICHUANG_KNOWLEDGE_SPACE_ID")
    if not SAFE_ID.fullmatch(tenant) or not SAFE_ID.fullmatch(space):
        raise ProxyError("tenant or knowledge-space configuration is invalid")
    return tenant, space


def _base_url() -> str:
    value = os.environ.get(
        "RUICHUANG_KNOWLEDGE_BASE_URL", "http://127.0.0.1:18877"
    ).strip().rstrip("/")
    parsed = parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProxyError("knowledge base URL must be HTTP or HTTPS")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise ProxyError("unencrypted HTTP is allowed only through a local tunnel")
    return value


def _keychain_token() -> str:
    service = os.environ.get("RUICHUANG_KNOWLEDGE_KEYCHAIN_SERVICE", "").strip()
    if not service:
        return ""
    account = os.environ.get("RUICHUANG_KNOWLEDGE_KEYCHAIN_ACCOUNT", "").strip()
    command = ["/usr/bin/security", "find-generic-password", "-w", "-s", service]
    if account:
        command.extend(["-a", account])
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _token() -> str:
    value = os.environ.get("RUICHUANG_KNOWLEDGE_API_TOKEN", "").strip()
    value = value or _keychain_token()
    if not value:
        raise ProxyError("enterprise knowledge credential is unavailable")
    return value


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {_token()}",
        **(extra or {}),
    }


def _request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    headers = _headers(extra_headers)
    data = body
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    endpoint = f"{_base_url()}{path}"
    req = request.Request(endpoint, data=data, headers=headers, method=method)
    effective_timeout = timeout or float(
        os.environ.get("RUICHUANG_KNOWLEDGE_TIMEOUT_SECONDS", "120")
    )
    try:
        with request.urlopen(req, timeout=effective_timeout) as response:
            raw = response.read()
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {}
        message = (
            (detail.get("error") or {}).get("message")
            or detail.get("detail")
            or f"HTTP {exc.code}"
        )
        raise ProxyError(f"knowledge API rejected the request: {message}") from exc
    except error.URLError as exc:
        raise ProxyError(f"knowledge API is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProxyError("knowledge API timed out") from exc
    try:
        decoded = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProxyError("knowledge API returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ProxyError("knowledge API returned an invalid response")
    return decoded


def _control_path(suffix: str = "") -> str:
    tenant, space = _scope()
    return (
        f"/control/v1/tenants/{parse.quote(tenant, safe='')}"
        f"/knowledge-spaces/{parse.quote(space, safe='')}{suffix}"
    )


def _safe_id(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_ID.fullmatch(normalized):
        raise ProxyError(f"{field} must be a stable ASCII identifier")
    return normalized


def _allowed_roots() -> tuple[Path, ...]:
    raw = os.environ.get("RUICHUANG_KNOWLEDGE_ALLOWED_ROOTS", "")
    candidates = [item for item in raw.split(os.pathsep) if item.strip()]
    if not candidates:
        candidates = [os.getcwd()]
    return tuple(Path(item).expanduser().resolve() for item in candidates)


def _manual_path(raw_path: Any) -> Path:
    candidate = Path(str(raw_path or "")).expanduser().resolve()
    if not candidate.is_file():
        raise ProxyError("manual is not a readable file")
    if not any(candidate == root or root in candidate.parents for root in _allowed_roots()):
        raise ProxyError("manual path is outside the approved roots")
    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ProxyError("manual must be .txt, .md, .pdf, or .docx")
    max_bytes = int(
        os.environ.get("RUICHUANG_KNOWLEDGE_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))
    )
    size = candidate.stat().st_size
    if size <= 0:
        raise ProxyError("manual is empty")
    if size > max_bytes:
        raise ProxyError(f"manual exceeds the {max_bytes}-byte limit")
    return candidate


def _authorized_path(raw_path: Any, *, directory: bool | None = None) -> Path:
    candidate = Path(str(raw_path or "")).expanduser().resolve()
    if directory is True and not candidate.is_dir():
        raise ProxyError("path is not a readable directory")
    if directory is False and not candidate.is_file():
        raise ProxyError("path is not a readable file")
    if not any(
        candidate == root or root in candidate.parents
        for root in _allowed_roots()
    ):
        raise ProxyError("path is outside the approved roots")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_hints(path: Path) -> dict[str, Any]:
    filename = path.name
    lowered = filename.lower()
    role = "manual"
    if any(token in lowered for token in ("policy", "warranty", "售后", "保修", "退换")):
        role = "policy"
    elif any(token in lowered for token in ("quick", "guide", "指南", "快速", "安装")):
        role = "guide"
    language = ""
    content = ""
    if path.suffix.lower() in {".txt", ".md"}:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = ""
    sample = content[:20000]
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in filename + sample)
    letters = sum(char.isascii() and char.isalpha() for char in filename + sample)
    if cjk and letters:
        language = "mixed"
    elif cjk:
        language = "zh-CN"
    elif letters:
        language = "en"
    version_match = re.search(
        r"(?i)(?:^|[-_ .])(v(?:er(?:sion)?)?[-_ ]?\d+(?:\.\d+){0,3}|\d{4}[-_.]\d{1,2}(?:[-_.]\d{1,2})?)(?:$|[-_ .])",
        filename,
    )
    model_candidates = sorted(
        set(
            re.findall(
                r"(?i)\b(?=[A-Z0-9-]{3,20}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b",
                filename,
            )
        )
    )
    image_references: list[str] = []
    if path.suffix.lower() == ".txt" and content:
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError:
            envelope = None
        if (
            isinstance(envelope, list)
            and len(envelope) == 2
            and isinstance(envelope[1], list)
        ):
            image_references = [
                str(item).strip()
                for item in envelope[1]
                if str(item).strip()
            ]
    policy_facts = {
        "warranty_period": sorted(
            set(
                re.findall(
                    r"(?i)(?:保修|质保|warranty).{0,24}?(\d+\s*(?:年|个月|月|years?|months?))",
                    content,
                )
            )
        ),
        "return_period": sorted(
            set(
                re.findall(
                    r"(?i)(\d+\s*(?:天|日|days?)).{0,24}?(?:退货|退换|return)",
                    content,
                )
            )
        ),
    }
    return {
        "role": role,
        "language": language,
        "version": version_match.group(1) if version_match else "",
        "model_candidates": model_candidates,
        "image_references": image_references,
        "policy_facts": {
            key: values for key, values in policy_facts.items() if values
        },
    }


def _analyze_directory(path: Path) -> dict[str, Any]:
    max_files = int(os.environ.get("RUICHUANG_KNOWLEDGE_MAX_PACKAGE_FILES", "2000"))
    all_files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in ALLOWED_EXTENSIONS | IMAGE_EXTENSIONS
        ),
        key=lambda item: str(item.relative_to(path)).lower(),
    )
    if not all_files:
        raise ProxyError("product folder contains no supported documents or images")
    if len(all_files) > max_files:
        raise ProxyError(f"product folder exceeds the {max_files}-file limit")
    documents: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = {}
    image_ids: set[str] = set()
    image_id_paths: dict[str, list[str]] = {}
    total_size = 0
    for item in all_files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total_size += size
        digest = _sha256(item)
        hashes.setdefault(digest, []).append(relative)
        if item.suffix.lower() in IMAGE_EXTENSIONS:
            image_ids.add(item.stem)
            image_id_paths.setdefault(item.stem, []).append(relative)
            images.append(
                {
                    "path": relative,
                    "image_id": item.stem,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
        else:
            documents.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                    **_document_hints(item),
                }
            )
    if not documents:
        raise ProxyError("product folder contains images but no supported document")
    duplicate_groups = [
        files for files in hashes.values() if len(files) > 1
    ]
    referenced_images = sorted(
        {
            image_id
            for document in documents
            for image_id in document["image_references"]
        }
    )
    missing_images = sorted(set(referenced_images) - image_ids)
    conflicts: list[dict[str, Any]] = []
    if duplicate_groups:
        conflicts.append(
            {
                "type": "duplicate_files",
                "groups": duplicate_groups,
            }
        )
    duplicate_image_ids = {
        image_id: paths
        for image_id, paths in image_id_paths.items()
        if len(paths) > 1
    }
    if duplicate_image_ids:
        conflicts.append(
            {
                "type": "duplicate_image_ids",
                "images": duplicate_image_ids,
            }
        )
    grouped_versions: dict[tuple[str, str], set[str]] = {}
    for document in documents:
        model = ",".join(document["model_candidates"])
        key = (document["role"], model)
        if document["version"]:
            grouped_versions.setdefault(key, set()).add(document["version"])
    for (role, model), versions in grouped_versions.items():
        if len(versions) > 1:
            conflicts.append(
                {
                    "type": "multiple_versions_same_role",
                    "role": role,
                    "model": model or None,
                    "versions": sorted(versions),
                }
            )
    if len(model_candidates := sorted(
        {
            model
            for document in documents
            for model in document["model_candidates"]
        }
    )) > 1:
        conflicts.append(
            {
                "type": "multiple_model_candidates",
                "models": model_candidates,
                "action": "confirm whether the package intentionally covers all models",
            }
        )
    policy_values: dict[str, set[str]] = {}
    for document in documents:
        for fact, values in document.get("policy_facts", {}).items():
            policy_values.setdefault(fact, set()).update(values)
    for fact, values in policy_values.items():
        if len(values) > 1:
            conflicts.append(
                {
                    "type": "policy_fact_conflict",
                    "fact": fact,
                    "values": sorted(values),
                }
            )
    if missing_images:
        conflicts.append(
            {
                "type": "missing_referenced_images",
                "image_ids": missing_images,
            }
        )
    model_candidates = sorted(model_candidates)
    return {
        "directory": str(path),
        "document_count": len(documents),
        "image_count": len(images),
        "total_size_bytes": total_size,
        "documents": documents,
        "images": images,
        "model_candidates": model_candidates,
        "referenced_image_ids": referenced_images,
        "missing_image_ids": missing_images,
        "conflicts": conflicts,
        "requires_review": bool(conflicts),
        "platform_boundary": (
            "The Agent host only analyzed and organized local files. "
            "Ruichuang will perform canonical parsing, chunking, image binding, indexing, and RAG."
        ),
    }


def _data(payload: dict[str, Any]) -> Any:
    return payload.get("data")


def _inspect(_: dict[str, Any]) -> dict[str, Any]:
    tenant, space = _scope()
    identity = _data(_request_json("GET", "/control/v1/me"))
    if not isinstance(identity, dict) or identity.get("tenant_id") != tenant:
        raise ProxyError("enterprise credential does not match the configured tenant")
    active = _data(_request_json("GET", _control_path("/active")))
    versions = _data(_request_json("GET", _control_path("/versions")))
    return {
        "tenant_id": tenant,
        "knowledge_space_id": space,
        "identity": identity,
        "active": active,
        "versions": versions or [],
    }


def _upload(arguments: dict[str, Any]) -> dict[str, Any]:
    path = _manual_path(arguments.get("source_path"))
    product_id = _safe_id(arguments.get("product_id"), "product_id")
    result = _request_json(
        "POST",
        _control_path("/ingestions"),
        body=path.read_bytes(),
        extra_headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(path.stat().st_size),
            "X-Filename": parse.quote(path.name, safe=""),
            "X-Product-Id": product_id,
        },
    )
    return _data(result)


def _analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    directory = _authorized_path(
        arguments.get("directory_path"),
        directory=True,
    )
    return _analyze_directory(directory)


def _stage_materials(arguments: dict[str, Any]) -> dict[str, Any]:
    directory = _authorized_path(
        arguments.get("directory_path"),
        directory=True,
    )
    product_id = _safe_id(arguments.get("product_id"), "product_id")
    product_name = str(arguments.get("product_name") or product_id).strip()[:256]
    analysis = _analyze_directory(directory)
    if any(
        item.get("type") == "duplicate_image_ids"
        for item in analysis["conflicts"]
    ):
        raise ProxyError(
            "product folder contains duplicate image IDs; rename or remove them before staging"
        )
    critical_conflicts = [
        item
        for item in analysis["conflicts"]
        if item.get("type")
        in {
            "multiple_versions_same_role",
            "multiple_model_candidates",
            "policy_fact_conflict",
        }
    ]
    conflict_resolution = str(
        arguments.get("conflict_resolution") or ""
    ).strip()
    if critical_conflicts and not conflict_resolution:
        raise ProxyError(
            "product folder has version/model/policy conflicts; obtain user confirmation and provide conflict_resolution before staging"
        )
    models = [
        str(item).strip()[:128]
        for item in arguments.get("models") or analysis["model_candidates"]
        if str(item).strip()
    ]
    manifest_files = []
    archive_files: list[tuple[Path, str]] = []
    for index, document in enumerate(analysis["documents"], start=1):
        source = directory / document["path"]
        archive_path = f"documents/{index:03d}-{source.name}"
        archive_files.append((source, archive_path))
        manifest_files.append(
            {
                "path": archive_path,
                "original_name": source.name,
                "source_relpath": document["path"],
                "role": document["role"],
                "language": document["language"],
                "model": (
                    document["model_candidates"][0]
                    if len(document["model_candidates"]) == 1
                    else ""
                ),
                "version": document["version"],
            }
        )
    for image in analysis["images"]:
        source = directory / image["path"]
        archive_files.append(
            (source, f"images/{source.name}")
        )
    package_manifest = {
        "schema_version": "1.0",
        "product_id": product_id,
        "product_name": product_name,
        "models": list(dict.fromkeys(models)),
        "files": manifest_files,
        "warnings": [
            str(item.get("type") or "")
            for item in analysis["conflicts"]
            if item.get("type")
        ],
        "agent_analysis": {
            "requires_review": analysis["requires_review"],
            "conflicts": analysis["conflicts"],
            "missing_image_ids": analysis["missing_image_ids"],
        },
        "conflict_resolution": conflict_resolution or None,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(package_manifest, ensure_ascii=False, indent=2),
        )
        for source, archive_path in archive_files:
            archive.write(source, archive_path)
    payload = buffer.getvalue()
    max_bytes = int(
        os.environ.get(
            "RUICHUANG_KNOWLEDGE_MAX_UPLOAD_BYTES",
            str(100 * 1024 * 1024),
        )
    )
    if len(payload) > max_bytes:
        raise ProxyError(
            f"compressed product package exceeds the {max_bytes}-byte limit"
        )
    result = _request_json(
        "POST",
        _control_path("/package-ingestions"),
        body=payload,
        extra_headers={
            "Content-Type": "application/zip",
            "Content-Length": str(len(payload)),
        },
    )
    staged = _data(result)
    return {
        **(staged if isinstance(staged, dict) else {"result": staged}),
        "agent_analysis": analysis,
    }


def _staging_status(arguments: dict[str, Any]) -> dict[str, Any]:
    staging_id = _safe_id(arguments.get("staging_id"), "staging_id")
    return _data(
        _request_json("GET", _control_path(f"/staging/{parse.quote(staging_id, safe='')}"))
    )


def _quality_check(arguments: dict[str, Any]) -> dict[str, Any]:
    staging_id = _safe_id(arguments.get("staging_id"), "staging_id")
    return _data(
        _request_json(
            "POST",
            _control_path(f"/staging/{parse.quote(staging_id, safe='')}/evaluate"),
            payload={},
        )
    )


def _diagnose(arguments: dict[str, Any]) -> dict[str, Any]:
    staging_id = _safe_id(arguments.get("staging_id"), "staging_id")
    return _data(
        _request_json(
            "GET",
            _control_path(
                f"/staging/{parse.quote(staging_id, safe='')}/diagnosis"
            ),
        )
    )


def _regression_cases(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    inline = arguments.get("cases")
    cases_path = str(arguments.get("cases_path") or "").strip()
    if inline is not None and cases_path:
        raise ProxyError("provide either cases or cases_path, not both")
    if cases_path:
        path = _authorized_path(cases_path, directory=False)
        if path.suffix.lower() != ".json":
            raise ProxyError("regression cases file must be JSON")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProxyError("regression cases file is invalid UTF-8 JSON") from exc
        inline = loaded.get("cases") if isinstance(loaded, dict) else loaded
    if not isinstance(inline, list) or not inline:
        raise ProxyError("regression cases or cases_path is required")
    if len(inline) > 500 or not all(isinstance(item, dict) for item in inline):
        raise ProxyError("regression cases must be 1-500 objects")
    return inline


def _run_regression(arguments: dict[str, Any]) -> dict[str, Any]:
    staging_id = _safe_id(arguments.get("staging_id"), "staging_id")
    return _data(
        _request_json(
            "POST",
            _control_path(
                f"/staging/{parse.quote(staging_id, safe='')}/regression"
            ),
            payload={"cases": _regression_cases(arguments)},
        )
    )


def _publish(arguments: dict[str, Any]) -> dict[str, Any]:
    staging_id = _safe_id(arguments.get("staging_id"), "staging_id")
    version = _safe_id(arguments.get("version"), "version")
    expected = f"PUBLISH {staging_id} AS {version}"
    supplied = str(arguments.get("confirmation_phrase") or "")
    if not hmac.compare_digest(supplied, expected):
        raise ProxyError(f"explicit approval required; confirmation_phrase must equal: {expected}")
    status = _staging_status({"staging_id": staging_id})
    if not isinstance(status, dict) or (status.get("quality") or {}).get("publishable") is not True:
        raise ProxyError("staged manual has not passed the ingestion quality check")
    return _data(
        _request_json(
            "POST",
            _control_path(f"/staging/{parse.quote(staging_id, safe='')}/publish"),
            payload={"version": version},
        )
    )


def _rollback(arguments: dict[str, Any]) -> dict[str, Any]:
    version = _safe_id(arguments.get("target_version"), "target_version")
    expected = f"ROLLBACK TO {version}"
    supplied = str(arguments.get("confirmation_phrase") or "")
    if not hmac.compare_digest(supplied, expected):
        raise ProxyError(f"explicit approval required; confirmation_phrase must equal: {expected}")
    snapshot = _inspect({})
    known = {str(item.get("version") or "") for item in snapshot["versions"]}
    if version not in known:
        raise ProxyError("target version is not present in the configured knowledge space")
    return _data(
        _request_json(
            "POST",
            _control_path(f"/versions/{parse.quote(version, safe='')}/rollback"),
            payload={},
        )
    )


def _audit(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(arguments.get("limit") or 100)
    if limit < 1 or limit > 500:
        raise ProxyError("limit must be between 1 and 500")
    tenant, _ = _scope()
    return _data(
        _request_json(
            "GET",
            f"/control/v1/tenants/{parse.quote(tenant, safe='')}/audit?limit={limit}",
        )
    )


def _verify(arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question") or "").strip()
    if not question:
        raise ProxyError("question is required")
    tenant, space = _scope()
    result = _request_json(
        "POST",
        "/tools/v1/answer_customer_question",
        payload={
            "schema_version": "1.0",
            "question": question,
            "attachments": [],
            "response_mode": "sync",
        },
        extra_headers={
            "X-Customer-Profile": "enterprise",
            "X-Tenant-Id": tenant,
            "X-Knowledge-Space": space,
            "X-Knowledge-Space-Id": space,
        },
    )
    return result


HANDLERS = {
    "inspect_knowledge_space": _inspect,
    "upload_product_manual": _upload,
    "analyze_product_materials": _analyze,
    "stage_product_materials": _stage_materials,
    "get_staging_status": _staging_status,
    "check_ingestion_quality": _quality_check,
    "diagnose_ingestion_blockers": _diagnose,
    "run_knowledge_regression": _run_regression,
    "publish_knowledge_version": _publish,
    "rollback_knowledge_version": _rollback,
    "list_audit_events": _audit,
    "verify_published_knowledge": _verify,
}


def _tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            }
        ],
        "structuredContent": payload if isinstance(payload, dict) else {"data": payload},
        "isError": is_error,
    }


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        protocol = requested if isinstance(requested, str) else PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {"name": name, **definition}
                    for name, definition in TOOLS.items()
                ]
            },
        }
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        handler = HANDLERS.get(name)
        if handler is None:
            result = _tool_result(
                {
                    "error_code": "permission_denied",
                    "message": "tool is not exposed by this knowledge-operations profile",
                    "retryable": False,
                },
                is_error=True,
            )
        else:
            try:
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise ProxyError("tool arguments must be an object")
                result = _tool_result(handler(arguments))
            except ProxyError as exc:
                result = _tool_result(
                    {
                        "error_code": "proxy_error",
                        "message": str(exc),
                        "retryable": False,
                    },
                    is_error=True,
                )
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> int:
    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            response = _handle(message)
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"invalid request: {exc}"},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
