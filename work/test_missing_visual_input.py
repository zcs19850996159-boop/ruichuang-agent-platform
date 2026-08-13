from __future__ import annotations

import threading

from agent_api import (
    AgentService,
    is_low_information_noise,
    normalize_route_question,
    requires_current_visual_input,
    requires_reference_context,
)
from conversation_memory import ConversationMemoryManager


class _DisabledStateStore:
    ready = False


class _NoRemoteMediaResult:
    cleaned_question = ""

    def image_items(self, *, start_index: int) -> list[dict[str, object]]:
        return []

    def context_text(self) -> str:
        return ""

    def public_metadata(self) -> dict[str, object]:
        return {"detected": 0, "usable": 0, "items": [], "errors": []}


class _NoRemoteMediaResolver:
    def resolve(self, question: str) -> _NoRemoteMediaResult:
        return _NoRemoteMediaResult()


def _service(tmp_path) -> AgentService:
    service = AgentService.__new__(AgentService)
    service.remote_media = _NoRemoteMediaResolver()
    service.memory = ConversationMemoryManager()
    service.memory_lock = threading.RLock()
    service.state_store = _DisabledStateStore()
    service.memory_path = tmp_path / "memory.json"
    service.question_route_hints = {}
    return service


def test_visual_reference_without_current_image_never_enters_manual_rag(tmp_path) -> None:
    service = _service(tmp_path)
    result = service.answer(
        {
            "question": "这张图片里的部件是什么？",
            "session_id": "missing-current-image",
        }
    )

    assert result["route"]["route_type"] == "clarification"
    assert result["route"]["manual_id"] == ""
    assert result["route"]["reason"] == "missing_visual_input"
    assert result["retrieval"]["mode"] == "pre_retrieval_clarification"
    assert result["input_images"]["received"] == 0
    assert result["images"] == []
    assert result["sources"] == []
    assert "上传" in result["answer"]
    assert "健身单车" not in result["answer"]


def test_missing_image_turn_clears_old_manual_and_followup_cannot_reuse_it(tmp_path) -> None:
    service = _service(tmp_path)
    memory = service.memory.get("old-manual-session")
    memory.active_route_type = "manual"
    memory.active_manual_id = "Manual14"
    memory.active_product = "健身单车"
    memory.last_user_question = "健身单车的部件有哪些？"

    first = service.answer(
        {
            "question": "这张图片里的部件是什么？",
            "session_id": "old-manual-session",
        }
    )
    state = service.memory.get("old-manual-session")
    assert first["route"]["reason"] == "missing_visual_input"
    assert state.active_route_type == "image_missing"
    assert state.active_manual_id == ""
    assert state.active_product == ""

    followup = service.answer(
        {
            "question": "那它怎么使用？",
            "session_id": "old-manual-session",
        }
    )
    assert followup["route"]["reason"] == "missing_visual_input_followup"
    assert followup["route"]["manual_id"] == ""
    assert "健身单车" not in followup["answer"]


def test_only_deictic_visual_requests_require_current_media() -> None:
    assert requires_current_visual_input("这张图片里的部件是什么？")
    assert requires_current_visual_input("What component is shown in this image?")
    assert not requires_current_visual_input("如何上传图片？")
    assert not requires_current_visual_input("健身单车的主要部件是什么？")


def test_invisible_characters_and_traditional_chinese_cannot_bypass_missing_image_guard(tmp_path) -> None:
    service = _service(tmp_path)
    for index, question in enumerate(
        (
            "这\u200b张\u200b图\u200b片里的部件是什么？",
            "這張圖片裡的部件是什麼？",
        )
    ):
        result = service.answer({"question": question, "session_id": f"noisy-image-{index}"})
        assert result["route"]["reason"] == "missing_visual_input"
        assert result["route"]["manual_id"] == ""
        assert result["images"] == []


def test_repeated_noise_is_rejected_and_clears_old_manual_context(tmp_path) -> None:
    service = _service(tmp_path)
    memory = service.memory.get("repeated-noise")
    memory.active_route_type = "manual"
    memory.active_manual_id = "Manual23"
    memory.active_product = "木工设备"
    memory.last_user_question = "如何调整刀片？"
    question = "噪声123!" * 300

    assert is_low_information_noise(question)
    result = service.answer({"question": question, "session_id": "repeated-noise"})
    assert result["route"]["reason"] == "low_information_input"
    assert result["route"]["manual_id"] == ""
    assert result["images"] == []
    state = service.memory.get("repeated-noise")
    assert state.active_route_type == "input_unclear"
    assert state.active_manual_id == ""
    assert state.active_product == ""
    followup = service.answer({"question": "那到底怎么办？", "session_id": "repeated-noise"})
    assert followup["route"]["reason"] == "low_information_input_followup"
    assert followup["route"]["manual_id"] == ""
    assert followup["images"] == []


def test_normal_long_question_is_not_classified_as_noise() -> None:
    question = (
        "设备在启动后显示错误码E01，同时伴随间歇性蜂鸣声。"
        "我已经检查了电源连接并重新安装电池，但问题仍然存在。"
        "请问还应该按照什么顺序检查传感器、连接线和控制面板？"
    )
    assert not is_low_information_noise(question)


def test_underspecified_reference_and_objectless_operation_are_detected() -> None:
    for question in (
        "这个部件是什么？",
        "这是什么部件？",
        "这个零件叫什么？",
        "它是什么型号？",
        "这个报错怎么处理？",
        "上面的按钮有什么用？",
        "刚才那个怎么安装？",
        "这里坏了怎么办？",
        "怎么更换？",
        "如何维修？",
        "这玩意怎么用？",
        "帮我看看这个是什么？",
    ):
        assert requires_reference_context(question), question


def test_underspecified_questions_stop_before_manual_rag(tmp_path) -> None:
    service = _service(tmp_path)
    questions = (
        "这个部件是什么？",
        "这是什么部件？",
        "这个零件叫什么？",
        "它是什么型号？",
        "这个报错怎么处理？",
        "上面的按钮有什么用？",
        "刚才那个怎么安装？",
        "这里坏了怎么办？",
        "怎么更换？",
        "如何维修？",
    )
    for index, question in enumerate(questions):
        result = service.answer({"question": question, "session_id": f"missing-ref-{index}"})
        assert result["route"]["reason"] == "missing_reference_context", question
        assert result["route"]["manual_id"] == ""
        assert result["retrieval"]["mode"] == "pre_retrieval_clarification"
        assert result["images"] == []
        assert result["sources"] == []


def test_valid_manual_followup_keeps_existing_context(tmp_path) -> None:
    service = _service(tmp_path)
    memory = service.memory.get("valid-followup")
    memory.active_route_type = "manual"
    memory.active_manual_id = "Manual17"
    memory.active_product = "冰箱"
    memory.last_user_question = "冰箱温度怎么设置？"
    assert requires_reference_context("这个按钮有什么用？")

    # The gate must allow a genuinely anchored follow-up to continue to normal routing.
    # The deliberately minimal service then reaches the next dependency instead of returning clarification.
    try:
        service.answer({"question": "这个按钮有什么用？", "session_id": "valid-followup"})
    except AttributeError as exc:
        assert "verified_visual_grounding" in str(exc)
    else:
        raise AssertionError("anchored follow-up was unexpectedly blocked by the reference gate")


def test_exact_official_question_bypasses_reference_gate(tmp_path) -> None:
    service = _service(tmp_path)
    service.question_route_hints = {normalize_route_question("怎么更换？"): "Manual27"}
    try:
        service.answer({"question": "怎么更换？", "session_id": "official-route"})
    except AttributeError as exc:
        assert "verified_visual_grounding" in str(exc)
    else:
        raise AssertionError("official routed question was unexpectedly blocked by the reference gate")
