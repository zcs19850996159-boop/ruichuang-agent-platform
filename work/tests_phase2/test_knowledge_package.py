from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from knowledge_lifecycle import KnowledgeLifecycleService, KnowledgeVersionStore
from knowledge_lifecycle import managed_retrieval
from knowledge_lifecycle.package import inspect_product_package


def write_package(
    path: Path,
    *,
    include_image: bool = True,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "product_id": "chair-x",
        "product_name": "Chair X",
        "models": ["CX100"],
        "files": [
            {
                "path": "documents/manual.txt",
                "role": "manual",
                "language": "zh-CN",
                "model": "CX100",
                "version": "v2",
            },
            {
                "path": "documents/policy.md",
                "role": "policy",
                "language": "zh-CN",
                "model": "CX100",
                "version": "v2",
            },
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False),
        )
        archive.writestr(
            "documents/manual.txt",
            json.dumps(
                [
                    "CX100 电池更换：关闭电源，打开电池仓。<PIC>",
                    ["CX100_battery"],
                ],
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "documents/policy.md",
            "# 保修政策\nCX100 电池保修一年。",
        )
        if include_image:
            archive.writestr("images/CX100_battery.png", b"synthetic-image")


def test_multi_document_package_runs_canonical_platform_pipeline(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "chair.zip"
    write_package(package_path)
    service = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "store")
    )

    staged = service.ingest_package(
        str(package_path),
        actor="alice",
        tenant_id="tenant-a",
        space_id="manuals",
    )

    manifest = staged["manifest"]
    assert manifest["schema_version"] == "1.1"
    assert manifest["package"]["document_count"] == 2
    assert len(
        [
            item
            for item in manifest["documents"]
            if item["product_id"] == "chair-x"
        ]
    ) == 2
    assert manifest["missing_referenced_images"] == []
    assert next(
        item
        for item in manifest["images"]
        if item["image_id"] == "CX100_battery"
    )["binding_status"] == "official_text_marker"

    regression = service.run_regression(
        staged["staging_id"],
        cases=[
            {
                "id": "battery",
                "question": "CX100 如何更换电池？",
                "expected_product_id": "chair-x",
                "required_terms": ["关闭电源"],
                "expected_image_ids": ["CX100_battery"],
            }
        ],
    )
    assert regression["status"] == "passed"
    report = service.evaluate(staged["staging_id"])
    if os.environ.get("KNOWLEDGE_REQUIRE_IMAGE_ENRICHMENT") == "1":
        assert report["publishable"] is False
        assert "image_enrichment_incomplete" in report["blockers"]
        with pytest.raises(ValueError, match="quality gate"):
            service.publish(
                staged["staging_id"],
                tenant_id="tenant-a",
                space_id="manuals",
                version="chair-v2",
                approved_by="alice",
            )
        return
    else:
        assert report["publishable"] is True
    assert report["regression"]["status"] == "passed"
    assert report["retrieval"]["strategy_version"] == "ruichuang-managed-hybrid-v2"
    assert report["retrieval"]["mode"] in {"hybrid", "lexical_fallback"}

    service.publish(
        staged["staging_id"],
        tenant_id="tenant-a",
        space_id="manuals",
        version="chair-v2",
        approved_by="alice",
    )
    search = service.search(
        "怎么更换电池",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    assert search["version"] == "chair-v2"
    assert search["retrieval"]["strategy_version"] == "ruichuang-managed-hybrid-v2"
    assert search["hits"][0]["product_id"] == "chair-x"


def test_missing_package_image_returns_exact_actionable_diagnosis(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "chair-missing.zip"
    write_package(package_path, include_image=False)
    service = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "store")
    )
    staged = service.ingest_package(
        str(package_path),
        actor="alice",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    report = service.evaluate(staged["staging_id"])
    assert report["publishable"] is False
    diagnosis = service.diagnose(staged["staging_id"])
    assert diagnosis["missing_image_ids"] == ["CX100_battery"]
    assert diagnosis["replacement_staging_required_after_file_changes"] is True
    assert diagnosis["next_action"] == "supply_exact_image_ids"


def test_regression_categorizes_staging_failures(tmp_path: Path) -> None:
    package_path = tmp_path / "chair.zip"
    write_package(package_path)
    service = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "store")
    )
    staged = service.ingest_package(
        str(package_path),
        actor="alice",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    report = service.run_regression(
        staged["staging_id"],
        cases=[
            {
                "id": "bad-expectations",
                "question": "CX100 电池",
                "expected_product_id": "other-product",
                "required_terms": ["并不存在的术语"],
                "expected_image_ids": ["missing-image"],
            }
        ],
    )
    assert report["status"] == "failed"
    assert report["failure_categories"] == {
        "image_missing": 1,
        "required_term_missing": 1,
        "wrong_product": 1,
    }


def test_package_rejects_path_traversal(tmp_path: Path) -> None:
    package_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "product_id": "unsafe",
                    "files": [
                        {
                            "path": "../manual.txt",
                            "role": "manual",
                        }
                    ],
                }
            ),
        )
        archive.writestr("../manual.txt", "unsafe")
    with pytest.raises(ValueError, match="unsafe package path"):
        inspect_product_package(package_path)


def test_managed_retriever_combines_vector_and_bm25_when_model_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "chunks.jsonl").write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in [
                {
                    "chunk_id": "battery",
                    "document_id": "manual",
                    "product_id": "chair-x",
                    "title": "CX100",
                    "section": "电池",
                    "text": "关闭电源后更换 CX100 电池。",
                    "image_ids": ["battery-image"],
                },
                {
                    "chunk_id": "cleaning",
                    "document_id": "manual",
                    "product_id": "chair-x",
                    "title": "CX100",
                    "section": "清洁",
                    "text": "使用软布清洁座椅。",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("MANAGED_HYBRID_EMBEDDING_MODEL", str(model))
    monkeypatch.setattr(
        managed_retrieval,
        "_encode_passages",
        lambda records, model_path, device: np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float16,
        ),
    )
    manifest = managed_retrieval.build_managed_index(snapshot)
    assert manifest["mode"] == "hybrid"
    monkeypatch.setattr(
        managed_retrieval.ManagedHybridRetriever,
        "_encode_query",
        lambda self, query: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    result = managed_retrieval.search_snapshot(
        snapshot,
        "CX100 电池怎么更换",
    )
    assert result["retrieval"]["mode"] == "hybrid"
    assert result["hits"][0]["chunk_id"] == "battery"
    assert result["hits"][0]["vector_score"] == 1.0


def test_section_context_recalls_bound_procedure_without_cross_section_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    chunks = [
        {
            "chunk_id": "self-test-heading",
            "document_id": "manual",
            "product_id": "wbtest01",
            "title": "manual",
            "section": "工程自检模式",
            "chunk_type": "heading",
            "text": "## 工程自检模式",
        },
        {
            "chunk_id": "self-test-procedure",
            "document_id": "manual",
            "product_id": "wbtest01",
            "title": "manual",
            "section": "工程自检模式",
            "chunk_type": "paragraph",
            "text": "长按橙色自检键 7 秒，蓝色指示灯闪烁三次。<PIC>",
            "image_ids": ["WBTEST01_0"],
        },
        {
            "chunk_id": "safety",
            "document_id": "manual",
            "product_id": "wbtest01",
            "title": "manual",
            "section": "安全提示",
            "chunk_type": "paragraph",
            "text": "漏水时不能进入工程自检模式。",
        },
    ]
    (snapshot / "chunks.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in chunks
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        managed_retrieval,
        "_model_configuration",
        lambda: ("", "cpu"),
    )

    managed_retrieval.build_managed_index(snapshot)
    result = managed_retrieval.search_snapshot(
        snapshot,
        "如何进入工程自检模式？",
        top_k=3,
    )

    hit_ids = [item["chunk_id"] for item in result["hits"]]
    assert "self-test-procedure" in hit_ids
    assert hit_ids.index("self-test-procedure") == hit_ids.index(
        "self-test-heading"
    ) + 1
    procedure = next(
        item
        for item in result["hits"]
        if item["chunk_id"] == "self-test-procedure"
    )
    assert procedure["image_ids"] == ["WBTEST01_0"]
    assert procedure["text"].startswith("长按橙色自检键 7 秒")
