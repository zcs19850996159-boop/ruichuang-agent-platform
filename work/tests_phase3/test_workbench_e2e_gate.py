from __future__ import annotations

from pathlib import Path


def test_workbench_e2e_gate_covers_user_visible_streaming_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    gate = (root / "e2e" / "workbench_streaming_gate.mjs").read_text(
        encoding="utf-8"
    )

    assert "Stream deltas differ from final answer" in gate
    assert "explicit_battery_question_after_context_prime" in gate
    assert "No manual image appeared before answer completion" in gate
    assert "All three images appeared only after final completion" in gate
    assert '["Manual27_1", "Manual27_2", "Manual27_3"]' in gate
    assert 'page.locator("#new-chat").click()' in gate
    assert "session_changed: true" in gate
    assert "sync_and_official_compatibility" in gate
    assert "rollback" in gate
    assert "CUSTOMER_SERVICE_API_TOKEN" not in gate


def test_workbench_e2e_runner_does_not_persist_credentials() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "e2e" / "run_workbench_streaming_gate.mjs").read_text(
        encoding="utf-8"
    )

    assert "process.env.CUSTOMER_SERVICE_API_TOKEN" in runner
    assert "token:" not in runner.split("process.stdout.write", 1)[1]
    assert "JSON.stringify(report" in runner


def test_codex_browser_gate_has_equivalent_inline_and_session_assertions() -> None:
    root = Path(__file__).resolve().parents[1]
    gate = (root / "e2e" / "workbench_browser_client_gate.mjs").read_text(
        encoding="utf-8"
    )

    assert "context_override_protection: true" in gate
    assert "No manual image appeared before answer completion" in gate
    assert "All three images appeared only after final completion" in gate
    assert "three_images_before_final_state" in gate
    assert "session_changed: true" in gate
    assert "competition-kb-v1" in gate
    assert "人工客服" in gate


def test_server_api_gate_covers_stream_sync_official_and_rollback() -> None:
    root = Path(__file__).resolve().parents[1]
    gate = (root / "e2e" / "api_compatibility_gate.py").read_text(
        encoding="utf-8"
    )

    assert "deltas ==" in gate
    assert 'response_mode": "stream"' in gate
    assert 'response_mode": "sync"' in gate
    assert 'f"{base_url}/chat"' in gate
    assert "official_chat_constraint_pass" in gate
    assert "rollback" in gate
    assert "KAFU_API_TOKEN" in gate
