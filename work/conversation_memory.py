from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


POLICY_TOPICS = {
    "return_refund": ["退货", "退款", "退换货", "7天无理由", "七天无理由", "return", "refund"],
    "shipping_fee": ["运费", "邮费", "shipping fee", "freight"],
    "trial": ["试用", "试用期", "试用期间", "trial"],
    "warranty_repair": ["保修", "维修", "故障", "质量问题", "warranty", "repair", "fault"],
    "exchange": ["换货", "更换", "补发", "exchange", "replacement"],
    "invoice": ["发票", "invoice"],
    "logistics": ["物流", "快递", "包裹", "签收", "shipping", "delivery"],
}

PRODUCT_HINTS = {
    "空气净化器": "Manual03",
    "室内空气质量指示灯": "Manual03",
    "空气质量指示灯": "Manual03",
    "室内空气质量": "Manual03",
    "VR头显": "Manual38",
    "更换耳塞": "Manual38",
    "立体声耳机": "Manual38",
    "耳塞": "Manual38",
    "searching status": "Manual22",
    "handset searching": "Manual22",
    "t-rail mounting": "Manual33",
    "t rail mounting": "Manual33",
    "空调": "Manual01",
    "人体工学椅": "Manual02",
    "空气净化器": "Manual03",
    "吹风机": "Manual04",
    "蒸汽清洁机": "Manual05",
    "洗碗机": "Manual06",
    "电钻": "Manual11",
    "健身单车": "Manual14",
    "健身追踪器": "Manual16",
    "冰箱": "Manual17",
    "发电机": "Manual18",
    "功能键盘": "Manual21",
    "儿童电动摩托车": "Manual26",
    "蓝牙激光鼠标": "Manual27",
    "烤箱": "Manual28",
    "水泵": "Manual31",
    "温控器": "Manual36",
    "VR头显": "Manual38",
    "摩托艇": "Manual40",
    "coffee machine": "Manual07",
    "milk frother": "Manual07",
    "steam nozzle": "Manual07",
    "milk drinks": "Manual07",
    "multi-use pressure cooker and air fryer": "Manual30",
    "multi use pressure cooker and air fryer": "Manual30",
    "multi-use pressure cooker": "Manual30",
    "multi use pressure cooker": "Manual30",
    "pressure-cooking": "Manual30",
    "pressure cooking": "Manual30",
    "pressure cooker": "Manual30",
    "quick release button": "Manual30",
    "steam release valve": "Manual30",
    "float valve": "Manual30",
    "anti-block shield": "Manual30",
    "sealing ring": "Manual30",
    "condensation collector": "Manual30",
    "air fryer basket or tray": "Manual30",
    "air fryer": "Manual08",
    "boat": "Manual09",
    "sound system": "Manual09",
    "stereo system": "Manual09",
    "auxiliary input jack": "Manual09",
    "camera": "Manual10",
    "earphones": "Manual12",
    "ereader": "Manual13",
    "fax": "Manual15",
    "grill": "Manual19",
    "jet ski": "Manual20",
    "watercraft": "Manual20",
    "landline": "Manual22",
    "lawn mower": "Manual23",
    "microwave": "Manual24",
    "motherboard": "Manual25",
    "m.2 storage": "Manual25",
    "m.2 socket": "Manual25",
    "m.2": "Manual25",
    "ngff": "Manual25",
    "mounting screw": "Manual25",
    "vacuum": "Manual32",
    "security camera": "Manual33",
    "t-rail": "Manual33",
    "power the camera": "Manual33",
    "snowmobile": "Manual34",
    "television": "Manual35",
    "toothbrush": "Manual37",
    "washing machine": "Manual39",
}


@dataclass
class MemoryTurn:
    role: str
    text: str
    resolved_question: str = ""
    route_type: str = ""
    manual_id: str = ""
    topics: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    session_id: str
    turns: list[MemoryTurn] = field(default_factory=list)
    active_route_type: str = ""
    active_manual_id: str = ""
    active_product: str = ""
    active_policy_topics: list[str] = field(default_factory=list)
    last_user_question: str = ""
    last_resolved_question: str = ""


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def detect_topics(text: str) -> list[str]:
    low = str(text or "").lower()
    topics = []
    for topic, hints in POLICY_TOPICS.items():
        if any(hint.lower() in low for hint in hints):
            topics.append(topic)
    return topics


def detect_product(text: str) -> tuple[str, str]:
    low = str(text or "").lower()
    for product, manual_id in sorted(PRODUCT_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if product.lower() in low:
            return product, manual_id
    return "", ""


def is_casual_utterance(text: str) -> bool:
    q = compact(text).lower()
    q = re.sub(r"[\s,\u002c\u3002\u002e\uff0c\uff01\u0021\uff1f\u003f\uff5e\u007e]+", "", q)
    if not q:
        return False
    casual_exact = {
        "\u4f60\u597d",
        "\u60a8\u597d",
        "hello",
        "hi",
        "hey",
        "\u55e8",
        "\u54c8\u55bd",
        "\u5728\u5417",
        "\u5728\u4e0d\u5728",
        "\u8c22\u8c22",
        "\u611f\u8c22",
        "thanks",
        "thankyou",
        "ok",
        "\u597d\u7684",
    }
    if q in casual_exact:
        return True
    if detect_product(text)[1]:
        return False

    assistant_refs = {
        "\u4f60",
        "\u60a8",
        "\u52a9\u624b",
        "\u667a\u80fd\u4f53",
        "\u5ba2\u670d",
        "\u673a\u5668\u4eba",
        "bot",
        "assistant",
    }
    capability_markers = {
        "\u80fd\u5e2e\u6211\u4ec0\u4e48",
        "\u80fd\u5e2e\u52a9\u6211\u4ec0\u4e48",
        "\u53ef\u4ee5\u5e2e\u6211\u4ec0\u4e48",
        "\u4f1a\u5e2e\u6211\u4ec0\u4e48",
        "\u80fd\u505a\u4ec0\u4e48",
        "\u53ef\u4ee5\u505a\u4ec0\u4e48",
        "\u80fd\u5e72\u4ec0\u4e48",
        "\u53ef\u4ee5\u5e72\u4ec0\u4e48",
        "\u4f1a\u4ec0\u4e48",
        "\u6709\u4ec0\u4e48\u529f\u80fd",
        "\u529f\u80fd\u662f\u4ec0\u4e48",
        "\u600e\u4e48\u5e2e\u6211",
        "\u4ecb\u7ecd\u4e0b\u81ea\u5df1",
        "\u4ecb\u7ecd\u4e00\u4e0b\u81ea\u5df1",
        "\u662f\u8c01",
    }
    if any(ref in q for ref in assistant_refs) and any(marker in q for marker in capability_markers):
        return True
    if any(marker in q for marker in {"whatcanyoudo", "howcanyouhelp", "whatdoyoudo"}):
        return True
    return False


def is_explicit_standalone_question(text: str) -> bool:
    q = compact(text)
    low = q.lower()
    normalized = re.sub(r"[\s,，.。!！?？~～]+", "", low)
    if not normalized:
        return False
    if detect_product(q)[1]:
        return True

    contextual_prefixes = (
        "那",
        "这个",
        "那个",
        "它",
        "上述",
        "前面",
        "刚才",
        "继续",
        "还",
        "另外",
        "then",
        "this",
        "that",
        "it",
        "whatabout",
        "howabout",
        "also",
    )
    if any(normalized.startswith(prefix) for prefix in contextual_prefixes):
        return False

    manual_markers = ("按照手册", "根据手册", "手册中", "说明书", "manual")
    question_markers = ("如何", "怎么", "怎样", "步骤", "方法", "how", "steps")
    if any(marker in low for marker in manual_markers) and any(
        marker in low for marker in question_markers
    ):
        return True

    standalone_prefixes = ("如何", "怎么", "怎样", "howto")
    return len(normalized) >= 6 and any(
        normalized.startswith(prefix) for prefix in standalone_prefixes
    )


def is_followup(text: str) -> bool:
    q = compact(text)
    low = q.lower()
    if is_casual_utterance(q):
        return False
    if is_explicit_standalone_question(q):
        return False
    followup_markers = [
        "那",
        "这个",
        "它",
        "上述",
        "刚才",
        "前面",
        "这种情况",
        "还",
        "另外",
        "怎么更换",
        "怎么拆",
        "怎么安装",
        "怎么使用",
        "怎么用",
        "需要自己",
        "能更换吗",
        "可以吗",
        "多久",
        "how about",
        "what about",
        "also",
        "then",
        "do i still",
    ]
    return any(marker in low for marker in followup_markers)


class ConversationMemoryManager:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id=session_id)
        return self.sessions[session_id]

    def export_session(self, session_id: str) -> dict[str, Any]:
        return asdict(self.get(session_id))

    def import_session(self, session_id: str, raw: dict[str, Any]) -> SessionMemory:
        turns = [
            MemoryTurn(
                role=str(item.get("role") or ""),
                text=str(item.get("text") or ""),
                resolved_question=str(item.get("resolved_question") or ""),
                route_type=str(item.get("route_type") or ""),
                manual_id=str(item.get("manual_id") or ""),
                topics=[str(x) for x in (item.get("topics") or [])],
            )
            for item in (raw.get("turns") or [])
            if isinstance(item, dict)
        ]
        memory = SessionMemory(
            session_id=session_id,
            turns=turns[-20:],
            active_route_type=str(raw.get("active_route_type") or ""),
            active_manual_id=str(raw.get("active_manual_id") or ""),
            active_product=str(raw.get("active_product") or ""),
            active_policy_topics=[str(x) for x in (raw.get("active_policy_topics") or [])],
            last_user_question=str(raw.get("last_user_question") or ""),
            last_resolved_question=str(raw.get("last_resolved_question") or ""),
        )
        self.sessions[session_id] = memory
        return memory

    def resolve_user_question(
        self,
        session_id: str,
        question: str,
        *,
        inherit_active_context: bool = True,
    ) -> dict[str, Any]:
        memory = self.get(session_id)
        raw = compact(question)
        topics = detect_topics(raw)
        product, manual_id = detect_product(raw)
        resolved = raw
        followup = (
            inherit_active_context
            and bool(memory.last_user_question)
            and is_followup(raw)
        )
        explicit_standalone = bool(memory.last_user_question) and (
            is_explicit_standalone_question(raw)
        )

        if followup:
            if not product and (memory.active_product or memory.active_manual_id):
                product = memory.active_product
                manual_id = memory.active_manual_id
            merged_topics = list(dict.fromkeys([*memory.active_policy_topics, *topics]))
            anchor = memory.last_user_question or memory.last_resolved_question
            if merged_topics and anchor:
                topic_text = self._topic_summary(merged_topics)
                resolved = f"当前会话已讨论“{anchor}”。用户继续追问：{raw}。请结合{topic_text}上下文回答。"
            elif anchor:
                resolved = f"当前会话已讨论“{anchor}”。用户继续追问：{raw}。"
            if product:
                resolved = f"当前会话正在讨论产品“{product}”。{resolved}"
        elif explicit_standalone and not manual_id:
            memory.active_manual_id = ""
            memory.active_product = ""
            if not topics:
                memory.active_route_type = ""

        route_type = "policy_service" if topics or memory.active_route_type == "policy_service" and followup else ""
        if manual_id:
            route_type = "manual"

        if route_type == "policy_service":
            memory.active_policy_topics = list(dict.fromkeys([*memory.active_policy_topics, *topics]))
        if manual_id:
            memory.active_manual_id = manual_id
            memory.active_product = product
        if route_type:
            memory.active_route_type = route_type
        memory.last_user_question = raw
        memory.last_resolved_question = resolved
        memory.turns.append(
            MemoryTurn(
                role="user",
                text=raw,
                resolved_question=resolved,
                route_type=route_type,
                manual_id=manual_id,
                topics=topics,
            )
        )
        return {
            "session_id": session_id,
            "original_question": raw,
            "resolved_question": resolved,
            "route_type_hint": route_type,
            "manual_id_hint": manual_id,
            "product_hint": product,
            "policy_topics": memory.active_policy_topics,
            "is_followup": followup,
            "explicit_standalone": explicit_standalone,
        }

    def add_assistant_answer(self, session_id: str, answer: str) -> None:
        memory = self.get(session_id)
        memory.turns.append(MemoryTurn(role="assistant", text=compact(answer)[:1000]))

    def to_context(self, session_id: str, max_turns: int = 6) -> str:
        memory = self.get(session_id)
        lines = [
            f"active_route_type={memory.active_route_type}",
            f"active_manual_id={memory.active_manual_id}",
            f"active_product={memory.active_product}",
            f"active_policy_topics={memory.active_policy_topics}",
        ]
        for turn in memory.turns[-max_turns:]:
            if turn.role == "user":
                lines.append(f"user: {turn.resolved_question or turn.text}")
            else:
                lines.append(f"assistant: {turn.text[:240]}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({sid: asdict(mem) for sid, mem in self.sessions.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _topic_summary(topics: list[str]) -> str:
        label = {
            "return_refund": "退换货/退款",
            "shipping_fee": "运费承担",
            "trial": "试用规则",
            "warranty_repair": "故障维修/保修",
            "exchange": "换货/更换",
            "invoice": "发票",
            "logistics": "物流签收",
        }
        return "、".join(label.get(topic, topic) for topic in topics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default="demo")
    parser.add_argument("--questions", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/rag_agent/conversation_memory_demo.json")
    args = parser.parse_args()

    questions = args.questions or [
        "请问你们家的商品支持7天无理由退换货吗？",
        "需要自己承担运费吗？",
        "我想试用商品，但是试用期间商品出现故障，而且不是人为操作导致的，同时我想延长试用期限，请问可以吗？",
        "另外故障商品能更换吗？",
    ]
    manager = ConversationMemoryManager()
    resolved = [manager.resolve_user_question(args.session_id, question) for question in questions]
    manager.save(args.output)
    print(json.dumps({"resolved": resolved, "context": manager.to_context(args.session_id)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
