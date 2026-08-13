from __future__ import annotations

import pytest

from agent_api import is_generic_product_issue
from generate_own_answers import (
    policy_answer_with_source,
    reviewed_toothbrush_topic_answer,
)


@pytest.mark.parametrize(
    "question",
    [
        "设备无法启动，应该怎么排查？",
        "机器开不了机怎么办？",
        "产品没反应，如何排查？",
        "The device won't start. How should I troubleshoot it?",
        "This equipment is not working.",
    ],
)
def test_generic_product_startup_issue_requires_product_identity(question: str) -> None:
    assert is_generic_product_issue(question)


@pytest.mark.parametrize(
    "question",
    [
        "发电机无法启动，应该怎么排查？",
        "空调不能启动怎么办？",
        "The generator won't start.",
    ],
)
def test_named_product_startup_issue_is_not_generic(question: str) -> None:
    assert not is_generic_product_issue(question)


@pytest.mark.parametrize(
    ("question", "source", "required", "forbidden"),
    [
        ("我购买的商品，发票丢了，还能申请售后吗？", "lost_invoice_after_sales", "不会直接导致无法申请售后", "支持按订单开具发票"),
        ("请问你们的物流能加急配送吗？需要额外收费吗？多久能送达？", "expedited_delivery", "不能保证", "提供订单号、问题描述"),
        ("我想修改订单的付款方式，能修改吗？", "change_payment_method", "已经支付后", "提供订单号、问题描述"),
        ("售后换货后，新商品仍然存在和上次一样的质量问题，该怎么处理？", "repeat_defect_after_replacement", "升级处理", "质保期通常"),
        ("商品说明书缺失，而且没有合格证，担心是三无产品，要求退款赔偿。", "missing_manual_certificate", "不一定等同于", "15天"),
        ("收到的型号发错了，错发型号已经停产且无法使用，要求退款赔偿。", "wrong_discontinued_model", "错发型号", "原支付路径"),
        ("商品表面有无法清洗的污渍，影响使用，想申请换货。", "uncleanable_stain", "换货", "提供订单号、问题描述"),
        ("商品生产日期被涂改过，无法确认是否过期，要求退款赔偿。", "altered_production_date", "停止食用或使用", "平台也会按商品品类"),
    ],
)
def test_critical_after_sales_rules(
    question: str,
    source: str,
    required: str,
    forbidden: str,
) -> None:
    answer, actual_source = policy_answer_with_source(question)
    assert actual_source == f"policy_critical:{source}"
    assert required in answer
    assert forbidden not in answer


def test_toothbrush_cleaning_excludes_battery_adjacency() -> None:
    answer, images, source = reviewed_toothbrush_topic_answer(
        "What are the best practices for cleaning and maintaining an electric or manual toothbrush to ensure hygiene and longevity?"
    )
    assert source == "reviewed_manual37:cleaning"
    assert images == ["toothbrush0_08", "toothbrush0_09", "toothbrush0_10"]
    assert "battery" not in answer.lower()
    assert answer.count("<PIC>") == len(images)


def test_toothbrush_battery_status_is_explicit() -> None:
    answer, images, source = reviewed_toothbrush_topic_answer(
        "What indicators are displayed on the toothbrush handle to show the battery status when it's on the charger or not charged?"
    )
    assert source == "reviewed_manual37:battery_status"
    assert images == ["Manual37_22", "Manual37_23"]
    assert "blinks white" in answer
    assert "bottom of the handle" in answer
    assert answer.count("<PIC>") == len(images)


def test_toothbrush_storage_requires_dryness() -> None:
    answer, images, source = reviewed_toothbrush_topic_answer(
        "What are the best practices for storing an electric or manual toothbrush when it is not in use to maintain hygiene and prolong its lifespan?"
    )
    assert source == "reviewed_manual37:storage"
    assert images == []
    assert "air-dry completely" in answer
    assert "do not seal a wet brush" in answer
    assert "battery" not in answer.lower()
