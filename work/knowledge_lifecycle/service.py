from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .chunking import build_chunks
from .extractors import extract_document
from .image_enrichment import enrich_stage_images
from .managed_retrieval import build_managed_index, search_snapshot
from .package import IMAGE_EXTENSIONS, extract_product_package
from .security import find_prompt_injection, inspect_upload
from .store import KnowledgeVersionStore, utc_now


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class KnowledgeLifecycleService:
    def __init__(
        self,
        store: KnowledgeVersionStore | None = None,
        *,
        reference_image_roots: list[str | Path] | None = None,
    ) -> None:
        self.store = store or KnowledgeVersionStore()
        configured_roots = reference_image_roots
        if configured_roots is None:
            configured_roots = [
                item
                for item in os.environ.get(
                    "KNOWLEDGE_REFERENCE_IMAGE_ROOTS",
                    "",
                ).split(os.pathsep)
                if item.strip()
            ]
        self.reference_image_roots = [
            Path(item).expanduser().resolve()
            for item in configured_roots
            if Path(item).expanduser().is_dir()
        ]

    def ingest(
        self,
        source_path: str,
        *,
        product_id: str,
        actor: str,
        tenant_id: str,
        space_id: str,
        original_name: str | None = None,
    ) -> dict[str, Any]:
        inspection = inspect_upload(source_path)
        staging_id = f"stg-{uuid.uuid4().hex[:16]}"
        stage = self.store.staging_path(staging_id)
        stage.mkdir(parents=True, exist_ok=False)
        source_dir = stage / "source"
        image_dir = stage / "images"
        source_dir.mkdir()
        image_dir.mkdir()
        active = self.store.read_active(tenant_id, space_id)
        base_version = str((active or {}).get("version") or "")
        base_documents: list[dict[str, Any]] = []
        base_chunks: list[dict[str, Any]] = []
        base_images: list[dict[str, Any]] = []
        base_warnings: list[str] = []
        replacing_product = False
        if base_version:
            base_path = self.store.version_path(tenant_id, space_id, base_version)
            base_manifest = json.loads(
                (base_path / "manifest.json").read_text(encoding="utf-8")
            )
            base_documents = list(base_manifest.get("documents") or [])
            if not base_documents and base_manifest.get("document"):
                base_documents = [
                    {
                        **base_manifest["document"],
                        "product_id": base_manifest.get("product_id"),
                    }
                ]
            base_chunks = self._load_chunks(base_path / "chunks.jsonl")
            base_images = list(base_manifest.get("images") or [])
            base_warnings = list(base_manifest.get("warnings") or [])
            replacing_product = any(
                str(item.get("product_id") or "") == product_id
                for item in base_documents
            )

            kept_chunks = [
                chunk
                for chunk in base_chunks
                if str(chunk.get("product_id") or "") != product_id
            ]
            kept_chunk_ids = {
                str(chunk.get("chunk_id") or "") for chunk in kept_chunks
            }
            kept_document_ids = {
                str(chunk.get("document_id") or "") for chunk in kept_chunks
            }
            removed_document_ids = {
                str(item.get("document_id") or "")
                for item in base_documents
                if str(item.get("document_id") or "") not in kept_document_ids
            }
            base_documents = [
                item
                for item in base_documents
                if str(item.get("document_id") or "") in kept_document_ids
            ]
            kept_images = []
            for image in base_images:
                if (
                    str(image.get("product_id") or "") == product_id
                    or str(image.get("document_id") or "") in removed_document_ids
                ):
                    continue
                bound = {
                    str(item) for item in image.get("bound_chunk_ids") or []
                }
                if bound and not bound.intersection(kept_chunk_ids):
                    continue
                source_file = base_path / str(image.get("file") or "")
                if source_file.is_file():
                    target_file = stage / str(image["file"])
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    kept_images.append(image)
            for base_document in base_documents:
                relative = str(base_document.get("source_file") or "")
                source_file = base_path / relative
                if relative and source_file.is_file():
                    target_file = stage / relative
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
            base_chunks = kept_chunks
            base_images = kept_images

        document_id = f"doc-{inspection.sha256[:16]}"
        copied = source_dir / f"{document_id}{inspection.extension}"
        shutil.copy2(inspection.path, copied)
        document = extract_document(copied)
        if original_name:
            document.title = Path(original_name).stem
        injection_findings = find_prompt_injection(document.text)
        new_chunks = build_chunks(
            document,
            document_id=document_id,
            product_id=product_id,
        )
        image_manifest: list[dict[str, Any]] = list(base_images)
        for index, (name, payload, source_page) in enumerate(document.images, start=1):
            suffix = Path(name).suffix.lower() or ".bin"
            image_id = f"{document_id}-image-{index}"
            path = image_dir / f"{image_id}{suffix}"
            path.write_bytes(payload)
            bound_chunks = [
                chunk["chunk_id"]
                for chunk in new_chunks
                if source_page is not None and chunk.get("page") == source_page
            ]
            for chunk in new_chunks:
                if chunk["chunk_id"] in bound_chunks:
                    chunk.setdefault("image_ids", []).append(image_id)
            image_manifest.append(
                {
                    "image_id": image_id,
                    "file": str(path.relative_to(stage)),
                    "source_name": name,
                    "source_page": source_page,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bound_chunk_ids": bound_chunks,
                    "binding_status": "auto_page" if bound_chunks else "requires_review",
                    "document_id": document_id,
                    "product_id": product_id,
                }
            )
        missing_referenced_images: list[str] = []
        for image_id in document.image_references:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", image_id):
                missing_referenced_images.append(image_id)
                continue
            matches: list[Path] = []
            for root in self.reference_image_roots:
                matches.extend(
                    candidate
                    for candidate in root.glob(f"{image_id}.*")
                    if candidate.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
                )
            source_image = sorted(set(matches), key=lambda item: str(item))[0] if matches else None
            bound_chunks = [
                chunk["chunk_id"]
                for chunk in new_chunks
                if image_id in (chunk.get("image_ids") or [])
            ]
            if source_image is None:
                missing_referenced_images.append(image_id)
                image_manifest.append(
                    {
                        "image_id": image_id,
                        "file": None,
                        "source_name": image_id,
                        "source_page": None,
                        "sha256": None,
                        "bound_chunk_ids": bound_chunks,
                        "binding_status": "missing_source_asset",
                        "document_id": document_id,
                        "product_id": product_id,
                    }
                )
                continue
            target_image = image_dir / f"{image_id}{source_image.suffix.lower()}"
            shutil.copy2(source_image, target_image)
            image_manifest.append(
                {
                    "image_id": image_id,
                    "file": str(target_image.relative_to(stage)),
                    "source_name": source_image.name,
                    "source_page": None,
                    "sha256": hashlib.sha256(target_image.read_bytes()).hexdigest(),
                    "bound_chunk_ids": bound_chunks,
                    "binding_status": (
                        "official_text_marker"
                        if bound_chunks
                        else "requires_review"
                    ),
                    "document_id": document_id,
                    "product_id": product_id,
                }
            )
        chunks = [*base_chunks, *new_chunks]
        with (stage / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
        document_manifest = {
            "document_id": document_id,
            "product_id": product_id,
            "original_name": Path(original_name).name if original_name else inspection.path.name,
            "source_file": str(copied.relative_to(stage)),
            "sha256": inspection.sha256,
            "mime": inspection.mime,
            "size_bytes": inspection.size_bytes,
            "parser": document.parser,
            "pages": len(document.pages),
            "tables": document.tables,
            "image_reference_count": len(document.image_references),
        }
        warnings = [*base_warnings, *document.warnings]
        if missing_referenced_images:
            warnings.append("missing_referenced_images")
        manifest = {
            "schema_version": "1.0",
            "staging_id": staging_id,
            "created_at": utc_now(),
            "created_by": actor,
            "tenant_id": tenant_id,
            "knowledge_space_id": space_id,
            "product_id": product_id,
            "base_version": base_version or None,
            "update_mode": (
                "replace_product"
                if replacing_product
                else "append_product"
            ),
            "document": document_manifest,
            "documents": [*base_documents, document_manifest],
            "chunks": len(chunks),
            "images": image_manifest,
            "missing_referenced_images": missing_referenced_images,
            "security": {
                "virus_scan": inspection.virus_scan,
                "prompt_injection_findings": injection_findings,
                "document_content_is_untrusted_data": True,
            },
            "warnings": warnings,
        }
        manifest["image_enrichment"] = enrich_stage_images(
            stage,
            chunks,
            image_manifest,
            product_id=product_id,
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.append_audit(
            "ingest",
            {
                "staging_id": staging_id,
                "actor": actor,
                "tenant_id": tenant_id,
                "knowledge_space_id": space_id,
                "source_sha256": inspection.sha256,
            },
        )
        return {
            "staging_id": staging_id,
            "status": "quarantined" if injection_findings else "staged",
            "manifest": manifest,
        }

    def ingest_package(
        self,
        package_path: str,
        *,
        actor: str,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        """Ingest one product folder as an isolated, canonical knowledge snapshot."""
        extraction_root = (
            self.store.root
            / "uploads"
            / f".package-{uuid.uuid4().hex[:16]}"
        )
        try:
            package = extract_product_package(package_path, extraction_root)
            first = package.documents[0]
            result = self.ingest(
                str(extraction_root / first.path),
                product_id=package.product_id,
                actor=actor,
                tenant_id=tenant_id,
                space_id=space_id,
                original_name=first.original_name,
            )
            staging_id = str(result["staging_id"])
            stage = self.store.staging_path(staging_id)
            source_dir = stage / "source"
            image_dir = stage / "images"
            manifest_path = stage / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks = self._load_chunks(stage / "chunks.jsonl")
            documents = list(manifest.get("documents") or [])
            images = list(manifest.get("images") or [])
            warnings = list(manifest.get("warnings") or [])
            injection_findings = list(
                (manifest.get("security") or {}).get(
                    "prompt_injection_findings"
                )
                or []
            )

            package_images: dict[str, Path] = {}
            duplicate_image_names: list[str] = []
            for relative in package.image_paths:
                source_image = extraction_root / relative
                image_id = source_image.stem
                existing = package_images.get(image_id)
                if existing is not None and existing.read_bytes() != source_image.read_bytes():
                    duplicate_image_names.append(image_id)
                    continue
                package_images[image_id] = source_image
            if duplicate_image_names:
                raise ValueError(
                    "package contains conflicting image assets with the same ID: "
                    + ", ".join(sorted(set(duplicate_image_names)))
                )

            def document_metadata(item: Any) -> dict[str, str]:
                return {
                    "role": item.role,
                    "language": item.language,
                    "model": item.model,
                    "version": item.version,
                }

            first_document_id = str((manifest.get("document") or {}).get("document_id") or "")
            first_metadata = document_metadata(first)
            for document_manifest in documents:
                if str(document_manifest.get("document_id") or "") == first_document_id:
                    document_manifest.update(first_metadata)
                    document_manifest["package_path"] = first.path
                    document_manifest["source_relpath"] = first.source_relpath
            if isinstance(manifest.get("document"), dict):
                manifest["document"].update(first_metadata)
                manifest["document"]["package_path"] = first.path
                manifest["document"]["source_relpath"] = first.source_relpath
            for chunk in chunks:
                if str(chunk.get("document_id") or "") == first_document_id:
                    chunk.update(first_metadata)

            known_document_hashes = {
                str(item.get("sha256") or "") for item in documents
            }
            duplicate_documents: list[str] = []

            def candidate_image(image_id: str) -> Path | None:
                packaged = package_images.get(image_id)
                if packaged is not None:
                    return packaged
                matches: list[Path] = []
                for root in self.reference_image_roots:
                    matches.extend(
                        candidate
                        for candidate in root.glob(f"{image_id}.*")
                        if candidate.suffix.lower() in IMAGE_EXTENSIONS
                    )
                return (
                    sorted(set(matches), key=lambda item: str(item))[0]
                    if matches
                    else None
                )

            def bind_reference(
                *,
                image_id: str,
                document_id: str,
                document_chunks: list[dict[str, Any]],
            ) -> None:
                bound_chunk_ids = [
                    str(chunk["chunk_id"])
                    for chunk in document_chunks
                    if image_id in (chunk.get("image_ids") or [])
                ]
                existing = next(
                    (
                        item
                        for item in images
                        if str(item.get("image_id") or "") == image_id
                    ),
                    None,
                )
                source_image = candidate_image(image_id)
                if source_image is None:
                    if existing is None:
                        images.append(
                            {
                                "image_id": image_id,
                                "file": None,
                                "source_name": image_id,
                                "source_page": None,
                                "sha256": None,
                                "bound_chunk_ids": bound_chunk_ids,
                                "binding_status": "missing_source_asset",
                                "document_id": document_id,
                                "document_ids": [document_id],
                                "product_id": package.product_id,
                            }
                        )
                    else:
                        existing["bound_chunk_ids"] = sorted(
                            set(existing.get("bound_chunk_ids") or [])
                            | set(bound_chunk_ids)
                        )
                    return
                target_image = image_dir / f"{image_id}{source_image.suffix.lower()}"
                if not target_image.is_file():
                    shutil.copy2(source_image, target_image)
                payload_hash = hashlib.sha256(target_image.read_bytes()).hexdigest()
                if existing is None:
                    images.append(
                        {
                            "image_id": image_id,
                            "file": str(target_image.relative_to(stage)),
                            "source_name": source_image.name,
                            "source_page": None,
                            "sha256": payload_hash,
                            "bound_chunk_ids": bound_chunk_ids,
                            "binding_status": (
                                "official_text_marker"
                                if bound_chunk_ids
                                else "requires_review"
                            ),
                            "document_id": document_id,
                            "document_ids": [document_id],
                            "product_id": package.product_id,
                        }
                    )
                    return
                existing.update(
                    {
                        "file": str(target_image.relative_to(stage)),
                        "source_name": source_image.name,
                        "sha256": payload_hash,
                        "binding_status": (
                            "official_text_marker"
                            if set(existing.get("bound_chunk_ids") or [])
                            | set(bound_chunk_ids)
                            else "requires_review"
                        ),
                        "bound_chunk_ids": sorted(
                            set(existing.get("bound_chunk_ids") or [])
                            | set(bound_chunk_ids)
                        ),
                        "document_ids": sorted(
                            set(existing.get("document_ids") or [])
                            | {str(existing.get("document_id") or ""), document_id}
                            - {""}
                        ),
                    }
                )

            for image in list(images):
                if image.get("binding_status") == "missing_source_asset":
                    bind_reference(
                        image_id=str(image.get("image_id") or ""),
                        document_id=str(image.get("document_id") or first_document_id),
                        document_chunks=[
                            chunk
                            for chunk in chunks
                            if str(chunk.get("document_id") or "")
                            == str(image.get("document_id") or first_document_id)
                        ],
                    )

            for package_file in package.documents[1:]:
                source_path = extraction_root / package_file.path
                inspection = inspect_upload(source_path)
                if inspection.sha256 in known_document_hashes:
                    duplicate_documents.append(package_file.path)
                    continue
                known_document_hashes.add(inspection.sha256)
                document_id = f"doc-{inspection.sha256[:16]}"
                copied = source_dir / f"{document_id}{inspection.extension}"
                shutil.copy2(inspection.path, copied)
                document = extract_document(copied)
                document.title = Path(package_file.original_name).stem
                injection_findings.extend(find_prompt_injection(document.text))
                metadata = document_metadata(package_file)
                new_chunks = build_chunks(
                    document,
                    document_id=document_id,
                    product_id=package.product_id,
                )
                for chunk in new_chunks:
                    chunk.update(metadata)
                chunks.extend(new_chunks)

                for index, (name, payload, source_page) in enumerate(
                    document.images,
                    start=1,
                ):
                    suffix = Path(name).suffix.lower() or ".bin"
                    image_id = f"{document_id}-image-{index}"
                    path = image_dir / f"{image_id}{suffix}"
                    path.write_bytes(payload)
                    bound_chunk_ids = [
                        str(chunk["chunk_id"])
                        for chunk in new_chunks
                        if source_page is not None
                        and chunk.get("page") == source_page
                    ]
                    for chunk in new_chunks:
                        if chunk["chunk_id"] in bound_chunk_ids:
                            chunk.setdefault("image_ids", []).append(image_id)
                    images.append(
                        {
                            "image_id": image_id,
                            "file": str(path.relative_to(stage)),
                            "source_name": name,
                            "source_page": source_page,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bound_chunk_ids": bound_chunk_ids,
                            "binding_status": (
                                "auto_page"
                                if bound_chunk_ids
                                else "requires_review"
                            ),
                            "document_id": document_id,
                            "document_ids": [document_id],
                            "product_id": package.product_id,
                        }
                    )
                for image_id in document.image_references:
                    if not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                        image_id,
                    ):
                        continue
                    bind_reference(
                        image_id=image_id,
                        document_id=document_id,
                        document_chunks=new_chunks,
                    )
                document_manifest = {
                    "document_id": document_id,
                    "product_id": package.product_id,
                    "original_name": package_file.original_name,
                    "package_path": package_file.path,
                    "source_relpath": package_file.source_relpath,
                    "source_file": str(copied.relative_to(stage)),
                    "sha256": inspection.sha256,
                    "mime": inspection.mime,
                    "size_bytes": inspection.size_bytes,
                    "parser": document.parser,
                    "pages": len(document.pages),
                    "tables": document.tables,
                    "image_reference_count": len(document.image_references),
                    **metadata,
                }
                documents.append(document_manifest)
                warnings.extend(document.warnings)

            referenced_image_ids = {
                str(image_id)
                for chunk in chunks
                if str(chunk.get("product_id") or "") == package.product_id
                for image_id in (chunk.get("image_ids") or [])
            }
            known_image_ids = {
                str(item.get("image_id") or "") for item in images
            }
            for image_id, source_image in package_images.items():
                if image_id in known_image_ids or image_id in referenced_image_ids:
                    continue
                target_image = image_dir / f"{image_id}{source_image.suffix.lower()}"
                shutil.copy2(source_image, target_image)
                images.append(
                    {
                        "image_id": image_id,
                        "file": str(target_image.relative_to(stage)),
                        "source_name": source_image.name,
                        "source_page": None,
                        "sha256": hashlib.sha256(target_image.read_bytes()).hexdigest(),
                        "bound_chunk_ids": [],
                        "binding_status": "package_asset_requires_review",
                        "document_id": None,
                        "document_ids": [],
                        "product_id": package.product_id,
                    }
                )

            missing_referenced_images = sorted(
                {
                    str(item.get("image_id") or "")
                    for item in images
                    if item.get("binding_status") == "missing_source_asset"
                }
                - {""}
            )
            warnings = [
                warning
                for warning in warnings
                if warning != "missing_referenced_images"
            ]
            if missing_referenced_images:
                warnings.append("missing_referenced_images")
            if duplicate_documents:
                warnings.append("duplicate_package_documents_skipped")
            package_warnings = package.metadata.get("warnings") or []
            if isinstance(package_warnings, list):
                warnings.extend(
                    str(item)[:256]
                    for item in package_warnings
                    if str(item).strip()
                )

            manifest.update(
                {
                    "schema_version": "1.1",
                    "product_id": package.product_id,
                    "product_name": package.product_name,
                    "models": list(package.models),
                    "documents": documents,
                    "chunks": len(chunks),
                    "images": images,
                    "missing_referenced_images": missing_referenced_images,
                    "security": {
                        **(manifest.get("security") or {}),
                        "prompt_injection_findings": sorted(
                            set(injection_findings)
                        ),
                    },
                    "warnings": list(dict.fromkeys(warnings)),
                    "package": {
                        "schema_version": "1.0",
                        "document_count": len(package.documents),
                        "image_asset_count": len(package.image_paths),
                        "duplicate_documents_skipped": duplicate_documents,
                        "metadata": package.metadata,
                    },
                }
            )
            manifest["image_enrichment"] = enrich_stage_images(
                stage,
                chunks,
                images,
                product_id=package.product_id,
            )
            with (stage / "chunks.jsonl").open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(
                        json.dumps(
                            chunk,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.store.append_audit(
                "ingest_package",
                {
                    "staging_id": staging_id,
                    "actor": actor,
                    "tenant_id": tenant_id,
                    "knowledge_space_id": space_id,
                    "product_id": package.product_id,
                    "document_count": len(package.documents),
                    "image_asset_count": len(package.image_paths),
                },
            )
            return {
                "staging_id": staging_id,
                "status": (
                    "quarantined"
                    if injection_findings
                    else "staged"
                ),
                "manifest": manifest,
            }
        finally:
            if extraction_root.exists():
                shutil.rmtree(extraction_root)

    def evaluate(
        self,
        staging_id: str,
        regression_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = self.store.staging_path(staging_id)
        manifest_path = stage / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("staging package does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = self._load_chunks(stage / "chunks.jsonl")
        normalized = [re.sub(r"\s+", "", item["text"]).lower() for item in chunks]
        duplicate_count = sum(count - 1 for count in Counter(normalized).values() if count > 1)
        injection = manifest["security"]["prompt_injection_findings"]
        warnings = list(manifest.get("warnings") or [])
        scanned_pdf = "scanned_or_low_text_pdf" in warnings
        ocr_ready = not scanned_pdf or __import__("os").environ.get("KNOWLEDGE_OCR_PROVIDER", "").strip()
        blockers: list[str] = []
        if not chunks:
            blockers.append("no_text_chunks")
        if injection:
            blockers.append("prompt_injection_detected")
        if not ocr_ready:
            blockers.append("scanned_pdf_requires_ocr")
        if manifest.get("missing_referenced_images"):
            blockers.append("referenced_images_missing")
        enrichment = manifest.get("image_enrichment") or {}
        if (
            os.environ.get("KNOWLEDGE_REQUIRE_IMAGE_ENRICHMENT", "0") == "1"
            and (
                int(enrichment.get("enriched") or 0)
                < int(enrichment.get("total") or 0)
                or int(enrichment.get("requires_review") or 0) > 0
            )
        ):
            blockers.append("image_enrichment_incomplete")
        package_metadata = (
            (manifest.get("package") or {}).get("metadata") or {}
        )
        agent_analysis = package_metadata.get("agent_analysis") or {}
        critical_package_conflicts = [
            item
            for item in agent_analysis.get("conflicts") or []
            if isinstance(item, dict)
            and item.get("type")
            in {
                "multiple_versions_same_role",
                "multiple_model_candidates",
                "policy_fact_conflict",
            }
        ]
        if (
            critical_package_conflicts
            and not str(
                package_metadata.get("conflict_resolution") or ""
            ).strip()
        ):
            blockers.append("package_conflicts_require_review")
        require_regression = os.environ.get("KNOWLEDGE_REQUIRE_REGRESSION", "0") == "1"
        stored_regression_path = stage / "regression_report.json"
        if regression_report is not None:
            stored_regression_path.write_text(
                json.dumps(regression_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        stored_regression = (
            json.loads(stored_regression_path.read_text(encoding="utf-8"))
            if stored_regression_path.is_file()
            else None
        )
        regression = regression_report or stored_regression or {
            "status": "not_configured",
            "total": 0,
            "passed": 0,
            "failed": 0,
        }
        if int(regression.get("failed") or 0) > 0:
            blockers.append("knowledge_regression_failed")
        if require_regression and not stored_regression and not regression_report:
            blockers.append("knowledge_regression_required")
        retrieval = build_managed_index(stage)
        manifest["retrieval"] = retrieval
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if (
            os.environ.get("KNOWLEDGE_REQUIRE_VECTOR_INDEX", "0") == "1"
            and retrieval.get("mode") != "hybrid"
        ):
            blockers.append("managed_vector_index_required")
        documents = list(manifest.get("documents") or [manifest["document"]])
        report = {
            "schema_version": "1.0",
            "staging_id": staging_id,
            "evaluated_at": utc_now(),
            "publishable": not blockers,
            "blockers": blockers,
            "metrics": {
                "document_count": len(documents),
                "page_count": sum(int(item.get("pages") or 0) for item in documents),
                "chunk_count": len(chunks),
                "image_count": len(manifest.get("images") or []),
                "unbound_image_count": sum(
                    not item.get("bound_chunk_ids")
                    for item in manifest.get("images") or []
                ),
                "missing_image_count": len(
                    manifest.get("missing_referenced_images") or []
                ),
                "table_count": sum(int(item.get("tables") or 0) for item in documents),
                "image_enriched_count": int(enrichment.get("enriched") or 0),
                "image_ocr_ready_count": int(enrichment.get("ocr_ready") or 0),
                "image_review_required_count": int(enrichment.get("requires_review") or 0),
                "duplicate_chunk_ratio": round(duplicate_count / max(1, len(chunks)), 4),
                "average_chunk_chars": round(sum(len(item["text"]) for item in chunks) / max(1, len(chunks)), 2),
            },
            "warnings": warnings,
            "security": manifest["security"],
            "regression": regression,
            "retrieval": retrieval,
        }
        (stage / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.append_audit(
            "evaluate",
            {
                "staging_id": staging_id,
                "publishable": report["publishable"],
                "tenant_id": manifest["tenant_id"],
                "knowledge_space_id": manifest["knowledge_space_id"],
            },
        )
        return report

    def diagnose(self, staging_id: str) -> dict[str, Any]:
        stage = self.store.staging_path(staging_id)
        manifest_path = stage / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("staging package does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quality_path = stage / "quality_report.json"
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8"))
            if quality_path.is_file()
            else self.evaluate(staging_id)
        )
        recommendation_map = {
            "no_text_chunks": {
                "category": "document_unreadable",
                "action": "replace_or_ocr",
                "message": "Replace the empty/unsupported document or enable OCR for scanned pages.",
            },
            "prompt_injection_detected": {
                "category": "security_quarantine",
                "action": "manual_security_review",
                "message": "Keep the staging version quarantined and review the reported instruction-like content.",
            },
            "scanned_pdf_requires_ocr": {
                "category": "ocr_required",
                "action": "enable_ocr_or_replace_pdf",
                "message": "Enable the approved OCR provider or provide a searchable PDF/TXT export.",
            },
            "referenced_images_missing": {
                "category": "missing_assets",
                "action": "supply_exact_image_ids",
                "message": "Add the exact missing image IDs to the product folder and create a replacement staging package.",
            },
            "knowledge_regression_failed": {
                "category": "regression_failure",
                "action": "review_failed_cases",
                "message": "Review categorized regression failures before publishing.",
            },
            "knowledge_regression_required": {
                "category": "regression_required",
                "action": "run_regression",
                "message": "Run the knowledge-version regression suite before publishing.",
            },
            "managed_vector_index_required": {
                "category": "vector_index_unavailable",
                "action": "configure_embedding_model",
                "message": "Configure the approved local embedding model and rebuild the staging quality report.",
            },
            "package_conflicts_require_review": {
                "category": "material_conflict",
                "action": "resolve_version_model_policy_conflicts",
                "message": "Resolve and record the reported version, model, or policy conflicts before creating a replacement staging package.",
            },
        }
        blockers = list(quality.get("blockers") or [])
        diagnosis = {
            "schema_version": "1.0",
            "staging_id": staging_id,
            "publishable": quality.get("publishable") is True,
            "blockers": [
                {
                    "code": blocker,
                    **recommendation_map.get(
                        blocker,
                        {
                            "category": "unknown",
                            "action": "manual_review",
                            "message": "Review this blocker before publishing.",
                        },
                    ),
                }
                for blocker in blockers
            ],
            "missing_image_ids": list(
                manifest.get("missing_referenced_images") or []
            ),
            "unbound_images": [
                str(item.get("image_id") or "")
                for item in manifest.get("images") or []
                if not item.get("bound_chunk_ids")
                and item.get("binding_status") != "missing_source_asset"
            ],
            "document_warnings": list(manifest.get("warnings") or []),
            "next_action": (
                "request_explicit_publish_approval"
                if quality.get("publishable") is True
                else (
                    recommendation_map.get(blockers[0], {}).get("action")
                    if blockers
                    else "run_ingestion_quality_check"
                )
            ),
            "replacement_staging_required_after_file_changes": bool(
                set(blockers)
                & {
                    "no_text_chunks",
                    "scanned_pdf_requires_ocr",
                    "referenced_images_missing",
                }
            ),
        }
        self.store.append_audit(
            "diagnose",
            {
                "staging_id": staging_id,
                "tenant_id": manifest.get("tenant_id"),
                "knowledge_space_id": manifest.get("knowledge_space_id"),
                "blockers": blockers,
            },
        )
        return diagnosis

    def run_regression(
        self,
        staging_id: str,
        *,
        cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not cases or len(cases) > 500:
            raise ValueError("regression cases must contain between 1 and 500 cases")
        stage = self.store.staging_path(staging_id)
        manifest_path = stage / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("staging package does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        build_managed_index(stage)
        base_version = str(manifest.get("base_version") or "")
        baseline_path = (
            self.store.version_path(
                str(manifest.get("tenant_id") or ""),
                str(manifest.get("knowledge_space_id") or ""),
                base_version,
            )
            if base_version
            else None
        )
        baseline_index = (
            (
                baseline_path / "hybrid_index"
                if (baseline_path / "hybrid_index" / "manifest.json").is_file()
                else (
                    self.store.root
                    / "retrieval_cache"
                    / self.store.scope_key(
                        str(manifest.get("tenant_id") or ""),
                        str(manifest.get("knowledge_space_id") or ""),
                    )
                    / base_version
                    / "hybrid_index"
                )
            )
            if baseline_path is not None
            else None
        )

        def evaluate_case(
            case: dict[str, Any],
            snapshot: Path,
            *,
            index_dir: Path | None = None,
        ) -> tuple[bool, list[str], dict[str, Any]]:
            question = str(case.get("question") or "").strip()
            if not question:
                raise ValueError("every regression case must include a question")
            result = search_snapshot(
                snapshot,
                question,
                top_k=5,
                index_dir=index_dir,
            )
            hits = list(result.get("hits") or [])
            failures: list[str] = []
            if not result.get("evidence_sufficient") or not hits:
                failures.append("evidence_missing")
            expected_product = str(case.get("expected_product_id") or "").strip()
            if expected_product and (
                not hits
                or str(hits[0].get("product_id") or "") != expected_product
            ):
                failures.append("wrong_product")
            # Evaluate the same top-k evidence set returned to the managed
            # answer path. Restricting regression to three of five returned
            # hits caused false failures when a relevant same-page companion
            # was ranked fourth or fifth.
            evidence_text = "\n".join(
                str(item.get("text") or "") for item in hits
            ).lower()
            required_terms = [
                str(item).strip()
                for item in case.get("required_terms") or []
                if str(item).strip()
            ]
            missing_terms = [
                term for term in required_terms if term.lower() not in evidence_text
            ]
            if missing_terms:
                failures.append("required_term_missing")
            expected_images = {
                str(item).strip()
                for item in case.get("expected_image_ids") or []
                if str(item).strip()
            }
            actual_images = {
                str(image_id)
                for item in hits
                for image_id in item.get("image_ids") or []
            }
            missing_images = sorted(expected_images - actual_images)
            if missing_images:
                failures.append("image_missing")
            return (
                not failures,
                failures,
                {
                    "top_hits": [
                        {
                            "chunk_id": item.get("chunk_id"),
                            "product_id": item.get("product_id"),
                            "score": item.get("score"),
                            "image_ids": item.get("image_ids") or [],
                        }
                        for item in hits
                    ],
                    "missing_terms": missing_terms,
                    "missing_image_ids": missing_images,
                    "retrieval_mode": (
                        result.get("retrieval") or {}
                    ).get("mode"),
                },
            )

        case_results: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter()
        for index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                raise ValueError("every regression case must be an object")
            case_id = str(case.get("id") or f"case-{index}")
            passed, failures, details = evaluate_case(case, stage)
            baseline_passed: bool | None = None
            if baseline_path is not None and baseline_path.is_dir():
                baseline_passed, _, _ = evaluate_case(
                    case,
                    baseline_path,
                    index_dir=baseline_index,
                )
                if baseline_passed and not passed:
                    failures.append("regression_from_baseline")
            category_counts.update(failures)
            case_results.append(
                {
                    "id": case_id,
                    "question": str(case.get("question") or ""),
                    "passed": passed,
                    "failure_categories": failures,
                    "baseline_passed": baseline_passed,
                    **details,
                }
            )
        passed_count = sum(item["passed"] for item in case_results)
        report = {
            "schema_version": "1.0",
            "staging_id": staging_id,
            "run_at": utc_now(),
            "status": (
                "passed"
                if passed_count == len(case_results)
                else "failed"
            ),
            "baseline_version": base_version or None,
            "total": len(case_results),
            "passed": passed_count,
            "failed": len(case_results) - passed_count,
            "failure_categories": dict(sorted(category_counts.items())),
            "cases": case_results,
        }
        (stage / "regression_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.store.append_audit(
            "regression",
            {
                "staging_id": staging_id,
                "tenant_id": manifest.get("tenant_id"),
                "knowledge_space_id": manifest.get("knowledge_space_id"),
                "total": report["total"],
                "failed": report["failed"],
                "failure_categories": report["failure_categories"],
            },
        )
        return report

    def publish(
        self,
        staging_id: str,
        *,
        tenant_id: str,
        space_id: str,
        version: str,
        approved_by: str,
    ) -> dict[str, Any]:
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        return self.store.publish_staging(
            staging_id,
            tenant_id=tenant_id,
            space_id=space_id,
            version=version,
            approved_by=approved_by,
        )

    def rollback(
        self,
        *,
        tenant_id: str,
        space_id: str,
        target_version: str,
        actor: str,
    ) -> dict[str, Any]:
        return self.store.rollback(tenant_id, space_id, target_version, actor)

    def list_versions(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> list[dict[str, Any]]:
        return self.store.list_versions(tenant_id, space_id)

    def list_staging(
        self,
        *,
        tenant_id: str,
        space_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return path-free staging summaries scoped to one knowledge space."""
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        rows: list[dict[str, Any]] = []
        for stage in self.store.staging.iterdir():
            if not stage.is_dir():
                continue
            try:
                metadata = self.staging_metadata(stage.name)
            except ValueError:
                continue
            if (
                str(metadata.get("tenant_id") or "") != tenant_id
                or str(metadata.get("knowledge_space_id") or "") != space_id
            ):
                continue
            quality = metadata.get("quality") or {}
            regression_path = stage / "regression_report.json"
            regression = (
                json.loads(regression_path.read_text(encoding="utf-8"))
                if regression_path.is_file()
                else quality.get("regression") or {}
            )
            document = metadata.get("document") or {}
            rows.append(
                {
                    "staging_id": metadata["staging_id"],
                    "created_at": metadata.get("created_at"),
                    "created_by": metadata.get("created_by"),
                    "tenant_id": metadata.get("tenant_id"),
                    "knowledge_space_id": metadata.get("knowledge_space_id"),
                    "product_id": metadata.get("product_id"),
                    "document": {
                        "document_id": document.get("document_id"),
                        "original_name": document.get("original_name"),
                        "sha256": document.get("sha256"),
                        "mime": document.get("mime"),
                        "pages": document.get("pages"),
                    },
                    "chunk_count": metadata.get("chunk_count"),
                    "image_count": metadata.get("image_count"),
                    "status": metadata.get("status"),
                    "quality": {
                        "publishable": quality.get("publishable"),
                        "blockers": quality.get("blockers") or [],
                        "evaluated_at": quality.get("evaluated_at"),
                    },
                    "regression": {
                        "status": regression.get("status"),
                        "total": int(regression.get("total") or 0),
                        "passed": int(regression.get("passed") or 0),
                        "failed": int(regression.get("failed") or 0),
                    },
                }
            )
        rows.sort(
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("staging_id") or ""),
            ),
            reverse=True,
        )
        return rows[:limit]

    def active_version(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any] | None:
        active = self.store.read_active(tenant_id, space_id)
        if active is None:
            return None
        return {
            "tenant_id": active.get("tenant_id"),
            "knowledge_space_id": active.get("knowledge_space_id"),
            "version": active.get("version"),
            "previous_version": active.get("previous_version"),
            "activated_at": active.get("activated_at"),
            "actor": active.get("actor"),
        }

    def staging_metadata(self, staging_id: str) -> dict[str, Any]:
        """Return non-path staging metadata for authorization and release UIs."""
        stage = self.store.staging_path(staging_id)
        manifest_path = stage / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("staging package does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quality_path = stage / "quality_report.json"
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8"))
            if quality_path.is_file()
            else None
        )
        return {
            "staging_id": staging_id,
            "created_at": manifest.get("created_at"),
            "created_by": manifest.get("created_by"),
            "tenant_id": manifest.get("tenant_id"),
            "knowledge_space_id": manifest.get("knowledge_space_id"),
            "product_id": manifest.get("product_id"),
            "base_version": manifest.get("base_version"),
            "update_mode": manifest.get("update_mode"),
            "document": manifest.get("document") or {},
            "documents": manifest.get("documents") or [],
            "package": manifest.get("package"),
            "chunk_count": int(manifest.get("chunks") or 0),
            "image_count": len(manifest.get("images") or []),
            "status": (
                "blocked"
                if quality and quality.get("publishable") is not True
                else "evaluated"
                if quality
                else "staged"
            ),
            "quality": quality,
        }

    def search(
        self,
        question: str,
        *,
        tenant_id: str,
        space_id: str,
        version: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        active = self.store.read_active(tenant_id, space_id)
        selected_version = version or str((active or {}).get("version") or "")
        if not selected_version:
            raise ValueError("no active managed knowledge version")
        path = self.store.version_path(tenant_id, space_id, selected_version)
        index_dir = (
            path / "hybrid_index"
            if (path / "hybrid_index" / "manifest.json").is_file()
            else (
                self.store.root
                / "retrieval_cache"
                / self.store.scope_key(tenant_id, space_id)
                / selected_version
                / "hybrid_index"
            )
        )
        result = search_snapshot(
            path,
            question,
            top_k=top_k,
            index_dir=index_dir,
        )
        return {"version": selected_version, **result}

    def resolve_image(
        self,
        *,
        tenant_id: str,
        space_id: str,
        version: str,
        image_id: str,
    ) -> Path:
        version_path = self.store.version_path(tenant_id, space_id, version)
        manifest_path = version_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("knowledge version does not exist")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        match = next(
            (item for item in manifest.get("images") or [] if item.get("image_id") == image_id),
            None,
        )
        if not match:
            raise ValueError("knowledge image does not exist")
        candidate = (version_path / str(match["file"])).resolve()
        if version_path not in candidate.parents or not candidate.is_file():
            raise ValueError("invalid knowledge image path")
        return candidate

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
