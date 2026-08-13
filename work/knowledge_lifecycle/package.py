from __future__ import annotations

import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


PACKAGE_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"
DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_PACKAGE_ENTRIES = int(os.environ.get("KNOWLEDGE_PACKAGE_MAX_ENTRIES", "2000"))
MAX_PACKAGE_BYTES = int(
    os.environ.get("KNOWLEDGE_PACKAGE_MAX_UNCOMPRESSED_BYTES", str(500 * 1024 * 1024))
)


@dataclass(frozen=True, slots=True)
class PackageFile:
    path: str
    original_name: str
    source_relpath: str
    role: str
    language: str
    model: str
    version: str


@dataclass(frozen=True, slots=True)
class ProductKnowledgePackage:
    product_id: str
    product_name: str
    models: tuple[str, ...]
    documents: tuple[PackageFile, ...]
    image_paths: tuple[str, ...]
    metadata: dict[str, Any]


def _safe_member_path(raw: str) -> PurePosixPath:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe package path: {raw!r}")
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def _read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = {info.filename.replace("\\", "/"): info for info in archive.infolist()}
    info = names.get(MANIFEST_NAME)
    if info is None:
        raise ValueError("knowledge package must contain manifest.json at its root")
    if info.file_size > 1024 * 1024:
        raise ValueError("knowledge package manifest is too large")
    try:
        payload = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("knowledge package manifest must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("knowledge package manifest must be an object")
    return payload


def inspect_product_package(package_path: str | Path) -> ProductKnowledgePackage:
    source = Path(package_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("knowledge package does not exist")
    if not zipfile.is_zipfile(source):
        raise ValueError("knowledge package must be a valid ZIP archive")
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_PACKAGE_ENTRIES:
            raise ValueError("knowledge package entry count is outside the allowed range")
        total_size = 0
        member_names: set[str] = set()
        for info in infos:
            safe_path = _safe_member_path(info.filename)
            normalized = safe_path.as_posix()
            if normalized in member_names:
                raise ValueError(f"duplicate package member: {normalized}")
            member_names.add(normalized)
            if _is_symlink(info):
                raise ValueError(f"symbolic links are not allowed in packages: {normalized}")
            total_size += int(info.file_size)
            if total_size > MAX_PACKAGE_BYTES:
                raise ValueError("knowledge package expands beyond the configured size limit")

        manifest = _read_manifest(archive)
        if str(manifest.get("schema_version") or "") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported knowledge package schema: {manifest.get('schema_version')!r}"
            )
        product_id = str(manifest.get("product_id") or "").strip()
        if not SAFE_ID.fullmatch(product_id):
            raise ValueError("knowledge package product_id is invalid")
        product_name = str(manifest.get("product_name") or product_id).strip()[:256]
        raw_models = manifest.get("models") or []
        if not isinstance(raw_models, list) or not all(
            isinstance(item, str) for item in raw_models
        ):
            raise ValueError("knowledge package models must be a string array")
        models = tuple(dict.fromkeys(item.strip() for item in raw_models if item.strip()))

        raw_files = manifest.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("knowledge package files must be a non-empty array")
        documents: list[PackageFile] = []
        declared_paths: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ValueError("each knowledge package file must be an object")
            relative = _safe_member_path(str(raw_file.get("path") or "")).as_posix()
            if relative in declared_paths:
                raise ValueError(f"document is declared more than once: {relative}")
            declared_paths.add(relative)
            if relative not in member_names:
                raise ValueError(f"declared document is missing from package: {relative}")
            extension = Path(relative).suffix.lower()
            if extension not in DOCUMENT_EXTENSIONS:
                raise ValueError(f"unsupported document in package: {relative}")
            documents.append(
                PackageFile(
                    path=relative,
                    original_name=Path(
                        str(raw_file.get("original_name") or Path(relative).name)
                    ).name,
                    source_relpath=_safe_member_path(
                        str(raw_file.get("source_relpath") or relative)
                    ).as_posix(),
                    role=str(raw_file.get("role") or "manual").strip()[:64],
                    language=str(raw_file.get("language") or "").strip()[:32],
                    model=str(raw_file.get("model") or "").strip()[:128],
                    version=str(raw_file.get("version") or "").strip()[:128],
                )
            )

        image_paths = tuple(
            sorted(
                name
                for name in member_names
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS
            )
        )
        return ProductKnowledgePackage(
            product_id=product_id,
            product_name=product_name,
            models=models,
            documents=tuple(documents),
            image_paths=image_paths,
            metadata={
                key: value
                for key, value in manifest.items()
                if key not in {"files"}
            },
        )


def extract_product_package(
    package_path: str | Path,
    destination: str | Path,
) -> ProductKnowledgePackage:
    package = inspect_product_package(package_path)
    target_root = Path(destination).resolve()
    target_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(Path(package_path).expanduser().resolve()) as archive:
        for info in archive.infolist():
            safe_path = _safe_member_path(info.filename)
            if info.is_dir():
                (target_root / safe_path.as_posix()).mkdir(parents=True, exist_ok=True)
                continue
            target = (target_root / safe_path.as_posix()).resolve()
            if target_root not in target.parents:
                raise ValueError("knowledge package path escapes extraction root")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
    return package
