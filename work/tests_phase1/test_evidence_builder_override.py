from __future__ import annotations

import ast
from pathlib import Path
import unittest


WORK_DIR = Path(__file__).resolve().parents[1]


class EvidenceBuilderOverrideTests(unittest.TestCase):
    def test_online_selection_override_is_explicit_and_backward_compatible(self) -> None:
        tree = ast.parse((WORK_DIR / "build_own_evidence.py").read_text(encoding="utf-8"))
        build = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build"
            and any(arg.arg == "selection_override" for arg in node.args.kwonlyargs)
        )
        index = [arg.arg for arg in build.args.kwonlyargs].index("selection_override")
        self.assertIsInstance(build.args.kw_defaults[index], ast.Constant)
        self.assertIsNone(build.args.kw_defaults[index].value)

    def test_online_path_does_not_use_process_environment_bridge(self) -> None:
        source = (WORK_DIR / "agent_api.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_run_rag_once"
        )
        method_source = ast.get_source_segment(source, method) or ""
        self.assertNotIn("META_IMAGE_SELECTION_CACHE", method_source)
        self.assertNotIn("runtime_api_selector_", method_source)
        self.assertIn("selection_override=selected", method_source)
