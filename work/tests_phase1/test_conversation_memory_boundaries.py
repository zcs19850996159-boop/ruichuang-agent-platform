from __future__ import annotations

from conversation_memory import (
    ConversationMemoryManager,
    is_explicit_standalone_question,
    is_followup,
)


def test_explicit_manual_question_does_not_inherit_old_manual() -> None:
    memory = ConversationMemoryManager()
    first = memory.resolve_user_question(
        "session-a",
        "冰箱内照明灯不亮，如何按照手册更换灯泡？",
    )
    assert first["manual_id_hint"] == "Manual17"

    second = memory.resolve_user_question(
        "session-a",
        "如何按照手册更换电池？",
    )
    assert second["is_followup"] is False
    assert second["explicit_standalone"] is True
    assert second["manual_id_hint"] == ""
    assert second["resolved_question"] == "如何按照手册更换电池？"
    assert memory.get("session-a").active_manual_id == ""


def test_referential_short_question_still_inherits_context() -> None:
    memory = ConversationMemoryManager()
    memory.resolve_user_question(
        "session-b",
        "冰箱内照明灯不亮，如何按照手册更换灯泡？",
    )
    followup = memory.resolve_user_question("session-b", "那怎么拆下来？")

    assert followup["is_followup"] is True
    assert followup["explicit_standalone"] is False
    assert followup["manual_id_hint"] == "Manual17"
    assert "用户继续追问" in followup["resolved_question"]


def test_explicit_policy_question_does_not_inherit_product_context() -> None:
    memory = ConversationMemoryManager()
    memory.resolve_user_question(
        "session-c",
        "冰箱内照明灯不亮，如何按照手册更换灯泡？",
    )
    policy = memory.resolve_user_question("session-c", "如何申请退款？")

    assert policy["is_followup"] is False
    assert policy["route_type_hint"] == "policy_service"
    assert policy["manual_id_hint"] == ""


def test_standalone_detection_keeps_ambiguous_followups_conservative() -> None:
    assert is_explicit_standalone_question("如何按照手册更换电池？")
    assert is_explicit_standalone_question("How to replace the battery?")
    assert not is_explicit_standalone_question("那按照手册怎么更换？")
    assert not is_explicit_standalone_question("怎么更换？")
    assert not is_followup("如何按照手册更换电池？")
    assert is_followup("怎么更换？")


def test_short_unrelated_text_does_not_inherit_old_manual() -> None:
    memory = ConversationMemoryManager()
    memory.resolve_user_question("session-d", "空调滤网应该怎么清洁？")

    unrelated = memory.resolve_user_question("session-d", "土豆怎么做")

    assert unrelated["is_followup"] is False
    assert unrelated["manual_id_hint"] == ""
    assert unrelated["resolved_question"] == "土豆怎么做"


def test_new_image_boundary_can_disable_context_inheritance() -> None:
    memory = ConversationMemoryManager()
    memory.resolve_user_question("session-e", "空调滤网应该怎么清洁？")

    image_turn = memory.resolve_user_question(
        "session-e",
        "这是什么？",
        inherit_active_context=False,
    )

    assert image_turn["is_followup"] is False
    assert image_turn["manual_id_hint"] == ""
    assert image_turn["resolved_question"] == "这是什么？"


def test_visual_product_is_named_in_followup_context() -> None:
    memory = ConversationMemoryManager()
    state = memory.get("session-f")
    state.active_route_type = "image_understanding"
    state.active_product = "Logitech MX Master 3 鼠标"
    state.last_user_question = "这是什么？"

    followup = memory.resolve_user_question("session-f", "它怎么使用？")

    assert followup["is_followup"] is True
    assert "Logitech MX Master 3 鼠标" in followup["resolved_question"]
