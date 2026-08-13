from __future__ import annotations

from pathlib import Path

from customer_service_core.context import RequestContext
from customer_service_tools.service import CustomerServiceToolService
from knowledge_lifecycle import KnowledgeLifecycleService, KnowledgeVersionStore


class FakeApplication:
    def answer(self, payload, stream_callback=None, context=None):
        if stream_callback is not None:
            stream_callback("answer_delta", {"text": "请打开"})
            stream_callback("answer_delta", {"text": "电池仓。"})
        return {
            "answer": "请打开电池仓。",
            "images": [],
            "sources": [{"title": "Manual A", "page": 3}],
            "route": {"route_type": "manual"},
            "confidence": {"score": 0.88},
            "answer_check": {"pass": True},
        }


class FakeGateway:
    def generate(self, messages, **options):
        return "按住电源键三秒即可启动。"


def context(*, profile: str = "competition", role: str = "user") -> RequestContext:
    return RequestContext(
        request_id="req-1",
        trace_id="trace-1",
        profile=profile,
        tenant_id="tenant-a",
        knowledge_space_id="manuals",
        knowledge_version="competition-kb-v1",
        application_version="test",
        model_version="test",
        prompt_version="test",
        user_id="alice",
        role=role,
        permissions=("knowledge:write",) if role == "admin" else (),
    )


def test_realtime_answer_is_one_core_call_with_versioned_envelope(tmp_path: Path) -> None:
    service = CustomerServiceToolService(
        FakeApplication(),
        KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store")),
        FakeGateway(),
    )
    result = service.execute(
        "answer_customer_question",
        {"schema_version": "1.0", "question": "如何安装电池？"},
        context=context(),
    )
    assert result["schema_version"] == "1.0"
    assert result["tool"] == "answer_customer_question"
    assert result["data"]["answer"] == "请打开电池仓。"
    assert result["data"]["confidence"] == 0.88


def test_realtime_answer_forwards_native_stream_events(tmp_path: Path) -> None:
    service = CustomerServiceToolService(
        FakeApplication(),
        KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store")),
        FakeGateway(),
    )
    events = []
    result = service.execute(
        "answer_customer_question",
        {
            "schema_version": "1.0",
            "question": "如何安装电池？",
            "response_mode": "stream",
        },
        context=context(),
        stream_callback=lambda event, payload: events.append((event, payload)),
    )

    assert result["data"]["answer"] == "请打开电池仓。"
    assert [
        payload["text"]
        for event, payload in events
        if event == "answer_delta"
    ] == ["请打开", "电池仓。"]


def test_managed_version_is_retrieved_and_answered_with_one_model_call(tmp_path: Path) -> None:
    lifecycle = KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store"))
    manual = tmp_path / "x100.txt"
    manual.write_text("X100 启动方法\n按住电源键三秒即可启动设备。", encoding="utf-8")
    staged = lifecycle.ingest(
        str(manual),
        product_id="x100",
        actor="alice",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    lifecycle.evaluate(staged["staging_id"])
    lifecycle.publish(
        staged["staging_id"],
        tenant_id="tenant-a",
        space_id="manuals",
        version="knowledge-v1",
        approved_by="alice",
    )
    service = CustomerServiceToolService(FakeApplication(), lifecycle, FakeGateway())
    result = service.execute(
        "answer_customer_question",
        {"question": "X100 如何启动？"},
        context=context(profile="enterprise"),
    )
    assert result["data"]["answer"] == "按住电源键三秒即可启动。"
    assert result["data"]["knowledge_version"] == "knowledge-v1"
    assert result["data"]["evidence"]


def test_managed_answer_without_active_version_escalates_cleanly(
    tmp_path: Path,
) -> None:
    lifecycle = KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store"))
    service = CustomerServiceToolService(FakeApplication(), lifecycle, FakeGateway())

    result = service.execute(
        "answer_customer_question",
        {"question": "X100 如何启动？"},
        context=context(profile="enterprise"),
    )

    assert result["data"]["escalation_required"] is True
    assert result["data"]["knowledge_version"] == ""
    assert result["data"]["evidence"] == []
    assert result["data"]["validation"]["reason"] == "no_active_knowledge_version"
    assert "还没有已发布" in result["data"]["answer"]


def test_admin_permission_is_required(tmp_path: Path) -> None:
    service = CustomerServiceToolService(
        FakeApplication(),
        KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store")),
        FakeGateway(),
    )
    try:
        service.execute(
            "evaluate_knowledge_update",
            {"staging_id": "stg-nope"},
            context=context(role="user"),
        )
    except Exception as exc:
        assert getattr(exc, "http_status", None) == 403
    else:
        raise AssertionError("admin tool unexpectedly succeeded")


def test_validator_catches_pic_mismatch() -> None:
    service = CustomerServiceToolService(FakeApplication(), model_gateway=FakeGateway())
    result = service.execute(
        "validate_customer_answer",
        {"answer": "步骤如下 <PIC>", "images": [], "evidence": [{"page": 1}]},
        context=context(),
    )
    assert result["data"]["pass"] is False
    assert "PIC count" in result["data"]["issues"][0]


def test_admin_answer_audit_returns_batch_summary(tmp_path: Path) -> None:
    service = CustomerServiceToolService(
        FakeApplication(),
        KnowledgeLifecycleService(KnowledgeVersionStore(tmp_path / "store")),
        FakeGateway(),
    )
    result = service.execute(
        "audit_customer_answers",
        {
            "items": [
                {"id": "ok", "answer": "有依据的回答", "evidence": [{"page": 1}]},
                {"id": "bad", "answer": "错误图片 <PIC>", "images": [], "evidence": []},
            ]
        },
        context=context(role="admin"),
    )
    assert result["data"]["total"] == 2
    assert result["data"]["passed"] == 1
    assert result["data"]["failed"] == 1
