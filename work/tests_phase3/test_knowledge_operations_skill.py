from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "integrations" / "workbuddy" / "knowledge_operations"
PROXY_PATH = INTEGRATION / "scripts" / "knowledge_ops_proxy.py"


def load_proxy():
    spec = importlib.util.spec_from_file_location(
        "ruichuang_knowledge_operations_proxy",
        PROXY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KnowledgeOperationsSkillTests(unittest.TestCase):
    def test_exposes_separate_guarded_tool_profile(self) -> None:
        proxy = load_proxy()
        self.assertEqual(
            set(proxy.TOOLS),
            {
                "inspect_knowledge_space",
                "upload_product_manual",
                "analyze_product_materials",
                "stage_product_materials",
                "get_staging_status",
                "check_ingestion_quality",
                "diagnose_ingestion_blockers",
                "run_knowledge_regression",
                "publish_knowledge_version",
                "rollback_knowledge_version",
                "list_audit_events",
                "verify_published_knowledge",
            },
        )
        self.assertTrue(
            proxy.TOOLS["inspect_knowledge_space"]["annotations"]["readOnlyHint"]
        )
        self.assertTrue(
            proxy.TOOLS["publish_knowledge_version"]["annotations"][
                "destructiveHint"
            ]
        )
        self.assertTrue(
            proxy.TOOLS["rollback_knowledge_version"]["annotations"][
                "destructiveHint"
            ]
        )
        self.assertIn(
            "competition answer evaluation",
            proxy.TOOLS["check_ingestion_quality"]["description"],
        )
        self.assertIn(
            "staging-versus-active",
            proxy.TOOLS["run_knowledge_regression"]["description"],
        )

    def test_agent_analyzes_product_folder_without_reimplementing_rag(
        self,
    ) -> None:
        proxy = load_proxy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product"
            product.mkdir()
            (product / "CX100-manual-v2.txt").write_text(
                '["关闭电源后更换电池。<PIC>", ["CX100_battery"]]',
                encoding="utf-8",
            )
            (product / "CX100_battery.png").write_bytes(b"image")
            with patch.dict(
                os.environ,
                {"RUICHUANG_KNOWLEDGE_ALLOWED_ROOTS": str(root)},
            ):
                analysis = proxy._analyze(
                    {"directory_path": str(product)}
                )
        self.assertEqual(analysis["document_count"], 1)
        self.assertEqual(analysis["image_count"], 1)
        self.assertEqual(analysis["missing_image_ids"], [])
        self.assertIn("Ruichuang", analysis["platform_boundary"])

    def test_policy_conflicts_require_a_user_resolution_before_staging(
        self,
    ) -> None:
        proxy = load_proxy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product"
            product.mkdir()
            (product / "CX100-policy-v1.txt").write_text(
                "本产品保修 1 年。",
                encoding="utf-8",
            )
            (product / "CX100-policy-v2.txt").write_text(
                "本产品保修 2 年。",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"RUICHUANG_KNOWLEDGE_ALLOWED_ROOTS": str(root)},
            ), patch.object(
                proxy,
                "_request_json",
                side_effect=AssertionError("network must not be called"),
            ):
                analysis = proxy._analyze(
                    {"directory_path": str(product)}
                )
                self.assertTrue(
                    any(
                        item["type"] == "policy_fact_conflict"
                        for item in analysis["conflicts"]
                    )
                )
                with self.assertRaisesRegex(
                    proxy.ProxyError,
                    "provide conflict_resolution",
                ):
                    proxy._stage_materials(
                        {
                            "directory_path": str(product),
                            "product_id": "chair-x",
                        }
                    )

    def test_publish_and_rollback_require_exact_confirmation_before_network(
        self,
    ) -> None:
        proxy = load_proxy()
        with patch.object(
            proxy,
            "_request_json",
            side_effect=AssertionError("network must not be called"),
        ):
            with self.assertRaisesRegex(
                proxy.ProxyError,
                "explicit approval required",
            ):
                proxy._publish(
                    {
                        "staging_id": "stg-123",
                        "version": "knowledge-v2",
                        "confirmation_phrase": "yes",
                    }
                )
            with self.assertRaisesRegex(
                proxy.ProxyError,
                "explicit approval required",
            ):
                proxy._rollback(
                    {
                        "target_version": "knowledge-v1",
                        "confirmation_phrase": "yes",
                    }
                )

    def test_manual_paths_are_confined_to_authorized_roots(self) -> None:
        proxy = load_proxy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "allowed"
            denied = root / "denied"
            allowed.mkdir()
            denied.mkdir()
            good = allowed / "manual.txt"
            bad = denied / "manual.txt"
            good.write_text("产品操作步骤", encoding="utf-8")
            bad.write_text("其他资料", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RUICHUANG_KNOWLEDGE_ALLOWED_ROOTS": str(allowed)},
            ):
                self.assertEqual(proxy._manual_path(str(good)), good.resolve())
                with self.assertRaisesRegex(
                    proxy.ProxyError,
                    "outside the approved roots",
                ):
                    proxy._manual_path(str(bad))

    def test_plain_http_is_limited_to_local_tunnels(self) -> None:
        proxy = load_proxy()
        with patch.dict(
            os.environ,
            {"RUICHUANG_KNOWLEDGE_BASE_URL": "http://example.com"},
        ):
            with self.assertRaisesRegex(proxy.ProxyError, "local tunnel"):
                proxy._base_url()
        with patch.dict(
            os.environ,
            {"RUICHUANG_KNOWLEDGE_BASE_URL": "http://127.0.0.1:18877"},
        ):
            self.assertEqual(
                proxy._base_url(),
                "http://127.0.0.1:18877",
            )

    def test_workbuddy_skill_uses_ingestion_quality_language(self) -> None:
        instructions = (INTEGRATION / "SKILL.md").read_text(encoding="utf-8")
        config = (INTEGRATION / "skill.yml").read_text(encoding="utf-8")
        self.assertIn("入库质量检查", instructions)
        self.assertIn("不是比赛题目答案评测", instructions)
        self.assertIn("publish_knowledge_version", config)
        self.assertIn("rollback_knowledge_version", config)
        self.assertIn("explicit_approval", config)
        self.assertNotIn("rcp_", config)


if __name__ == "__main__":
    unittest.main()
