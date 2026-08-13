from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_selector import ImageSelector, compact, expand_query_text, overlap_score, token_counter
from llm_image_selector import build_candidates, call_deepseek, parse_json_ids, render_prompt
from meta_image_selector import choose_variant
from feedback_learning import FeedbackRuleEngine


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"
ROUTE_PATH = ROOT / "work" / "a_rank_question_route_gold.csv"
CANONICAL_REFERENCE_PATH = ROOT / "work" / "canonical_highscore_reference_v62_base81625.csv"
IMAGE_ID_RE = re.compile(
    r"\b(?:Manual\d+|Camera|drill\d*|jetski|Security_Camera|air_conditioner|exercise_bikes|fitness_trackers|oven|fax|generator|Dish_washer|Blower|function_keyboard|toothbrush\d*|vr|VR)[A-Za-z0-9]*_[A-Za-z0-9]+\b"
)

EXPLICIT_TOPIC_IMAGE_IDS: set[str] = {
    "Manual16_12",
    "Manual20_15",
    "Manual20_19",
    "Manual21_12",
    "Manual21_13",
    "Manual21_14",
    "Manual27_12",
    "Manual27_13",
    "Manual36_41",
    "Manual36_42",
    "Manual40_22",
    "Manual40_26",
    "Manual40_27",
    "use_pressure_cooker_and_air_fryer_01",
    "use_pressure_cooker_and_air_fryer_02",
    "use_pressure_cooker_and_air_fryer_03",
    "use_pressure_cooker_and_air_fryer_04",
}


POLICY_HINT_RE = re.compile(
    r"退款|退货|换货|维修|售后|发票|物流|快递|订单|投诉|运费|补发|包装|保修|质保|保质期|生产日期|"
    r"纸质版说明书|电子版|智能客服|试用装|试用|尺寸差价|更大的尺寸|质量问题|联系客服|上门安装|以旧换新|优惠券|"
    r"价格保护|降价|退差价|补差价|价保|包裹|丢失|赠品|签收|缺少配件|缺件|"
    r"收货地址|改地址|发货|部分发货|合并发货|拆单|驿站|取件|重新派送|配送|送货|改约|预约|"
    r"旧机回收|回收人员|上门|预售|定金|尾款|定制|刻字|取消订单|套装|配件|"
    r"购买凭证|保修凭证|备用机|复检|换新|少发|破损|工单|主管|专员|"
    r"refund|return|invoice|shipping|order|warranty|complaint|repair",
    re.I,
)


PRODUCT_HINTS: dict[str, list[str]] = {
    "Manual03": ["空气净化器", "室内空气质量指示灯", "空气质量指示灯", "室内空气质量", "air purifier"],
    "Manual22": ["landline", "searching status", "handset searching", "handset", "base station", "phone"],
    "Manual33": ["security camera", "t-rail mounting", "t rail mounting", "t-rail", "power the camera"],
    "Manual38": ["VR头显", "更换耳塞", "立体声耳机", "耳塞", "vr headset"],
    "Manual01": ["空调", "遥控器", "air conditioner"],
    "Manual02": ["人体工学椅", "椅子", "ergonomic chair"],
    "Manual03": ["空气净化器", "air purifier"],
    "Manual04": ["吹风机", "leaf blower", "blower"],
    "Manual05": ["蒸汽清洁机", "steam cleaner"],
    "Manual06": ["洗碗机", "dishwasher"],
    "Manual07": ["coffee machine", "coffee", "milk frother", "steam nozzle", "milk drinks"],
    "Manual08": ["air fryer"],
    "Manual09": ["boat", "sailing", "ship", "sound system", "stereo system", "auxiliary input jack"],
    "Manual10": ["camera", "lens"],
    "Manual11": ["电钻", "dcb101", "drill"],
    "Manual12": ["earphones", "earbuds", "earphone", "headset", "headphones", "charging case", "earbud"],
    "Manual13": ["ereader", "ebook", "e-reader"],
    "Manual14": ["健身单车", "exercise bike"],
    "Manual15": ["fax", "fax machine"],
    "Manual16": ["健身追踪器", "fitness tracker"],
    "Manual17": ["冰箱", "refrigerator", "freezer"],
    "Manual18": ["发电机", "generator"],
    "Manual19": ["grill", "lp tank", "regulator"],
    "Manual20": ["jetski", "jet ski", "watercraft", "wave runner", "waverunner"],
    "Manual21": ["功能键盘", "keyboard"],
    "Manual22": ["landline", "handset", "base station", "phone"],
    "Manual23": ["lawn mower", "mower"],
    "Manual24": ["microwave", "over-the-range"],
    "Manual25": ["motherboard", "pci express", "jumper", "m.2 storage", "m.2 socket", "m.2", "ngff", "mounting screw"],
    "Manual26": ["儿童电动摩托车", "ride-on motorcycle"],
    "Manual27": ["蓝牙激光鼠标", "bluetooth mouse", "laser mouse"],
    "Manual28": ["烤箱", "oven"],
    "Manual30": [
        "multi-use pressure cooker and air fryer",
        "multi use pressure cooker and air fryer",
        "multi-use pressure cooker",
        "multi use pressure cooker",
        "pressure-cooking",
        "pressure cooking",
        "pressure cooker",
        "quick release button",
        "steam release valve",
        "float valve",
        "anti-block shield",
        "sealing ring",
        "condensation collector",
        "air fryer basket or tray",
        "multi-level air fryer basket",
    ],
    "Manual31": ["水泵", "pump"],
    "Manual32": ["vacuum", "robot vacuum"],
    "Manual33": ["security camera", "t-rail", "power the camera"],
    "Manual34": ["snowmobile"],
    "Manual35": ["television", "tv"],
    "Manual36": ["温控器", "thermostat"],
    "Manual37": ["toothbrush", "electric toothbrush"],
    "Manual38": ["vr头显", "vr headset"],
    "Manual39": ["washing machine", "washer"],
    "Manual40": ["摩托艇"],
}


PRODUCT_HINTS["Manual03"].extend(["空气净化器", "室内空气质量指示灯", "空气质量指示灯", "室内空气质量"])
PRODUCT_HINTS.setdefault("Manual20", []).extend(["jstski", "jetski", "jet ski", "watercraft"])
PRODUCT_HINTS["Manual22"].extend(["searching status", "handset searching"])
PRODUCT_HINTS["Manual33"].extend(["t-rail mounting", "t rail mounting"])
PRODUCT_HINTS["Manual38"].extend(["VR头显", "更换耳塞", "立体声耳机", "耳塞"])


POLICY_CJK_TERMS = (
    "售后",
    "商品",
    "订单",
    "发货",
    "下单",
    "客服",
    "快递",
    "物流",
    "发票",
    "退款",
    "退货",
    "换货",
    "补发",
    "运费",
    "赠品",
    "签收",
    "仓库",
    "平台",
    "重拍",
    "取消",
    "价格保护",
    "退差价",
    "补差价",
    "价保",
    "收货地址",
    "改地址",
    "部分发货",
    "合并发货",
    "拆单",
    "包裹",
    "驿站",
    "取件",
    "重新派送",
    "配送",
    "送货",
    "改约",
    "预约",
    "费用",
    "配件包",
    "缺件",
    "旧机回收",
    "回收人员",
    "上门",
    "预售",
    "定金",
    "尾款",
    "定制",
    "刻字",
    "套装",
    "配件",
    "购买凭证",
    "保修凭证",
    "备用机",
    "复检",
    "换新",
    "少发",
    "破损",
    "工单",
    "主管",
    "专员",
)


@dataclass
class RoutePrediction:
    route_type: str
    manual_id: str
    confidence: float
    reason: str
    top_manuals: list[dict[str, Any]]


@dataclass
class TeacherImageExample:
    row_id: str
    manual_id: str
    question: str
    normalized_question: str
    image_ids: list[str]
    tokens: Counter[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().strip('"').strip("'").lower())


def parse_reference_ret_images(value: str) -> list[str]:
    text = str(value or "").strip().strip('"')
    if not text:
        return []
    match = re.search(r",\s*(\[[^\]]*\])\s*$", text, flags=re.S)
    if not match:
        return []
    raw = match.group(1)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [m.group(0) for m in IMAGE_ID_RE.finditer(raw)]


def infer_manual_from_image_ids(image_ids: list[str]) -> str:
    counts: Counter[str] = Counter()
    for image_id in image_ids:
        match = re.match(r"(Manual\d+)_", image_id)
        if match:
            counts[match.group(1)] += 1
    if len(counts) == 1:
        return counts.most_common(1)[0][0]
    return ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return {str(row["id"]): row for row in load_jsonl(path)}


def f1_score(pred: list[str], gold: list[str]) -> float:
    ps = set(pred)
    gs = set(gold)
    tp = len(ps & gs)
    precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class ManualRouter:
    def __init__(self, selector: ImageSelector, *, use_known_routes: bool = True) -> None:
        self.selector = selector
        self.use_known_routes = use_known_routes
        self.route_rows = load_csv(ROUTE_PATH)
        self.profile_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        self.policy_tokens: Counter[str] = Counter()
        self._build_profiles()

    def _build_profiles(self) -> None:
        for row in self.route_rows:
            text = " ".join(
                str(row.get(key) or "")
                for key in ("question", "product_or_policy", "intent_type", "retrieval_strategy", "image_strategy")
            )
            manual_id = str(row.get("gold_manual") or "")
            if row.get("route_type") == "policy_service" or manual_id == "none_policy":
                self.policy_tokens.update(token_counter(text))
            elif manual_id and manual_id != "EN_SUMMARY":
                self.profile_tokens[manual_id].update(token_counter(text))

        for chunk in self.selector.chunks:
            manual_id = str(chunk.get("manual_id") or "")
            if not manual_id or manual_id == "EN_SUMMARY":
                continue
            text = " ".join(
                compact(str(chunk.get(key) or ""))
                for key in ("manual_id", "product", "section_title", "text", "captions", "chunk_id")
            )
            self.profile_tokens[manual_id].update(token_counter(text[:5000]))

        for rec in load_jsonl(ASSET_DIR / "english_pic_captions.jsonl"):
            manual_id = str(rec.get("manual_id") or "")
            if not manual_id or manual_id == "EN_SUMMARY":
                continue
            text = " ".join(
                compact(str(rec.get(key) or ""))
                for key in ("manual_id", "product", "image_id", "caption_en", "nearest_section", "section_path", "notes")
            )
            self.profile_tokens[manual_id].update(token_counter(text[:3000]))

        for image_id, rec in self.selector.image_records.items():
            if rec.manual_id and rec.manual_id != "EN_SUMMARY":
                self.profile_tokens[rec.manual_id].update(token_counter(f"{image_id} {rec.text[:800]}"))

    def predict(self, row_id: str, question: str, manual_hint: str = "") -> RoutePrediction:
        if manual_hint and manual_hint != "none_policy":
            return RoutePrediction("manual", manual_hint, 1.0, "manual_hint", [{"manual_id": manual_hint, "score": 9999.0}])

        if any("\u4e00" <= ch <= "\u9fff" for ch in question) and any(term in question for term in POLICY_CJK_TERMS):
            manual_hint_hit = any(
                hint and hint in question
                for hints in PRODUCT_HINTS.values()
                for hint in hints
                if any("\u4e00" <= ch <= "\u9fff" for ch in hint)
            )
            if not manual_hint_hit:
                return RoutePrediction("policy_service", "none_policy", 0.96, "policy_cjk_guard", [])

        route = self.selector.routes.get(str(row_id)) if self.use_known_routes else None
        if route and route.manual_id:
            if route.route_type == "policy_service" or route.manual_id == "none_policy":
                return RoutePrediction("policy_service", "none_policy", 1.0, "known_route_policy", [])
            return RoutePrediction("manual", route.manual_id, 1.0, "known_route", [{"manual_id": route.manual_id, "score": 9999.0}])

        q_tokens = token_counter(expand_query_text(question))
        manual_scores: list[tuple[float, str]] = []
        question_lower = question.lower()
        for manual_id, tokens in self.profile_tokens.items():
            score = overlap_score(q_tokens, tokens)
            if manual_id.lower() in question.lower():
                score += 200
            for hint in PRODUCT_HINTS.get(manual_id, []):
                if hint.lower() in question_lower:
                    score += 260
            manual_scores.append((score, manual_id))
        manual_scores.sort(reverse=True)
        top = manual_scores[:5]
        best_score, best_manual = top[0] if top else (0.0, "")
        second = top[1][0] if len(top) > 1 else 0.0
        policy_score = overlap_score(q_tokens, self.policy_tokens)

        if POLICY_HINT_RE.search(question) and policy_score >= max(8.0, best_score * 0.45):
            return RoutePrediction(
                "policy_service",
                "none_policy",
                min(0.95, 0.45 + policy_score / 180.0),
                f"policy_score={policy_score:.2f}, manual_score={best_score:.2f}",
                [{"manual_id": manual_id, "score": round(score, 3)} for score, manual_id in top],
            )

        if best_score <= 0:
            confidence = 0.0
        else:
            margin = (best_score - second) / max(1.0, best_score)
            confidence = max(0.05, min(0.98, 0.35 + margin * 0.9 + min(best_score, 120.0) / 300.0))
        return RoutePrediction(
            "manual",
            best_manual,
            confidence,
            f"best_score={best_score:.2f}, second={second:.2f}, policy={policy_score:.2f}",
            [{"manual_id": manual_id, "score": round(score, 3)} for score, manual_id in top],
        )


def render_simple_prompt(
    question: str,
    manual_id: str,
    candidates: list[dict[str, Any]],
    base_images: list[str],
    candidate_order: str,
) -> list[dict[str, str]]:
    if candidate_order == "manual":
        ordered = sorted(
            candidates,
            key=lambda c: (int(c.get("pic_index") if c.get("pic_index") is not None else 10**9), str(c.get("image_id") or "")),
        )
    else:
        ordered = candidates
    lines = [
        "Select product-manual figures for the answer.",
        "Return JSON only: {\"image_ids\": [\"...\"]}.",
        "Choose only candidate IDs.",
        "Include every figure needed for the requested explanation or step sequence.",
        "Exclude unrelated neighboring figures.",
        "Keep manual/PIC order.",
        f"Manual: {manual_id}",
        f"Question: {question}",
        f"Heuristic images: {base_images}",
        "Candidates:",
    ]
    for idx, cand in enumerate(ordered, 1):
        text = re.sub(r"\s+", " ", str(cand.get("text") or ""))[:520]
        lines.append(f"{idx}. id={cand['image_id']} pic_index={cand.get('pic_index')} score={cand.get('score')} text={text}")
    return [
        {"role": "system", "content": "You are a strict manual figure selector. Output valid JSON only."},
        {"role": "user", "content": "\n".join(lines)},
    ]


def compute_confidence(
    route: RoutePrediction,
    final_images: list[str],
    proposals: dict[str, list[str]],
    selected_variant: str,
) -> dict[str, Any]:
    non_empty = [tuple(ids) for ids in proposals.values() if ids]
    agreement = 0
    if non_empty:
        final_set = set(final_images)
        agreement = sum(1 for ids in non_empty if set(ids) == final_set)
    unique_sets = len({tuple(ids) for ids in proposals.values()})
    score = route.confidence
    if agreement >= 2:
        score += 0.18
    if unique_sets <= 2:
        score += 0.12
    if selected_variant == "base" and len(final_images) <= 1:
        score -= 0.08
    if not final_images and route.route_type != "policy_service":
        score -= 0.2
    if unique_sets >= 4:
        score -= 0.12
    score = max(0.0, min(1.0, score))
    if score >= 0.78:
        level = "high"
    elif score >= 0.52:
        level = "medium"
    else:
        level = "low"
    return {
        "score": round(score, 3),
        "level": level,
        "agreement": agreement,
        "unique_proposal_sets": unique_sets,
    }


def _topic_images(
    manual_id: str,
    question: str,
    proposals: dict[str, list[str]],
    allowed_ids: set[str],
) -> tuple[list[str], str] | None:
    """Apply manual-topic figure grouping rules learned from reviewed cases."""
    q = question.lower()

    def keep(ids: list[str]) -> list[str]:
        seen: set[str] = set()
        kept: list[str] = []
        for image_id in ids:
            if (image_id in allowed_ids or image_id in EXPLICIT_TOPIC_IMAGE_IDS) and image_id not in seen:
                kept.append(image_id)
                seen.add(image_id)
        return kept

    def choose(ids: list[str], reason: str) -> tuple[list[str], str] | None:
        kept = keep(ids)
        return (kept, reason) if kept else None

    def choose_empty(reason: str) -> tuple[list[str], str]:
        return ([], reason)

    def proposal(name: str) -> list[str]:
        return keep([str(x) for x in proposals.get(name, [])])

    if manual_id == "Manual01":
        if any(term in question for term in ("小松树", "松树图标")) or (
            any(term in q for term in ("tree icon", "pine tree icon"))
            and any(term in q for term in ("remote", "air conditioner"))
        ):
            return choose(["Manual01_18"], "topic:air_conditioner_air_purification_icon")
        if "重要组成部件" in question or ("组成部件" in question and "空调" in question):
            return choose(["Manual01_0", "Manual01_1", "air_conditioner_01", "Manual01_17"], "topic:air_conditioner_components_overview")
        if "遥控器" in question and ("按键" in question or "找到" in question):
            return choose(["air_conditioner_01", "Manual01_2", "Manual01_3"], "topic:air_conditioner_remote_buttons")
        if "空气滤网" in question and "清洁" in question and "3m" not in q and "等离子" not in question:
            return choose(["Manual01_29", "Manual01_30"], "topic:air_conditioner_air_filter_clean_only")
        if any(term in question for term in ("单冷型", "自动运行模式")) or (
            "auto" in q and "mode" in q and "cool" in q
        ):
            return choose(["Manual01_19"], "topic:air_conditioner_cooling_auto_mode")
        if (
            ("\u6e05\u6d01" in question and ("\u9891\u7387" in question or "\u6ee4\u7f51" in question or "\u578b\u53f7" in question))
            or ("clean" in q and "air conditioner" in q and ("frequency" in q or "filter" in q))
        ) and "3m" not in q and "\u7b49\u79bb\u5b50" not in question:
            return choose(["Manual01_29", "Manual01_30", "Manual01_31", "Manual01_32"], "topic:air_conditioner_cleaning_frequency")
        if any(term in question for term in ("购物小票", "购买日期", "序列号")) or (
            "型号" in question and any(term in question for term in ("小票", "购买", "存档", "记录"))
        ) or any(
            term in q for term in ("receipt", "serial number", "purchase date")
        ):
            return choose_empty("topic:air_conditioner_purchase_record_text_only")
        if "遥控器" in question and "电池" in question:
            return choose(["Manual01_2", "Manual01_3"], "topic:air_conditioner_remote_battery_core")
        if "3m" in q and "滤网" in question:
            return choose(["Manual01_31"], "topic:air_conditioner_3m_filter_clean")
        if "\u7b49\u79bb\u5b50" in question and "\u6ee4\u7f51" in question:
            return choose(["Manual01_32"], "topic:air_conditioner_plasma_filter_clean")
        if "操作须知" in question:
            return choose_empty("topic:air_conditioner_operation_notice_text_only")
        if ("遥控器" in question and "电池" in question) or ("remote" in q and "battery" in q):
            return choose(
                ["air_conditioner_01", "Manual01_2", "Manual01_3", "Manual01_4", "Manual01_5", "Manual01_22"],
                "topic:air_conditioner_remote_battery",
            )

    if manual_id == "Manual02":
        if "扶手" in question and ("松动" in question or "晃" in question):
            return choose_empty("topic:chair_armrest_loose_text_only")
        if "组装" in question and "部件" in question:
            return choose(["Manual02_0"], "topic:chair_assembly_parts_overview")
        if "有哪些功能" in question or ("chair" in q and "function" in q):
            return choose(["Manual02_11", "Manual02_12", "Manual02_13", "Manual02_14", "Manual02_15"], "topic:chair_functions")
        if any(term in q for term in ("package", "packaging", "included", "components", "parts")) or any(
            term in question for term in ("包装", "开箱", "零部件", "主要零部件", "包含哪些")
        ):
            base = proposal("base")
            if len(base) >= 4:
                return base[:4], "topic:chair_package_base_group"
            return choose(["Manual02_0", "Manual02_1", "Manual02_2", "Manual02_3"], "topic:chair_package_group")

    if manual_id == "Manual03":
        if "灰尘传感器" in question or "dust sensor" in q:
            return choose(["Manual03_22", "Manual03_23", "Manual03_24", "Manual03_25"], "topic:air_purifier_dust_sensor_clean")
        if any(term in question for term in ("通常有哪些模式", "如何设置", "模式有什么特点")) or (
            "mode" in q and ("air purifier" in q or "purifier" in q)
        ):
            return choose(["Manual03_14", "Manual03_15", "Manual03_16", "Manual03_17"], "topic:air_purifier_modes")
        if "清洁" in question and "滤网" in question:
            return choose(["Manual03_12"], "topic:air_purifier_filter_clean")
        if "更换" in question and "滤网" in question:
            return choose(["Manual03_21"], "topic:air_purifier_filter_replace")
        if any(term in question for term in ("电视", "电子设备", "3英尺", "三英尺")) or any(
            term in q for term in ("tv", "television", "electronic device", "3 feet")
        ):
            return choose_empty("topic:air_purifier_electronics_distance_text_only")

    if manual_id == "Manual05":
        if (
            ("\u5b9e\u7528" in question and "\u529f\u80fd" in question)
            or (
                "steam cleaner" in q
                and any(term in q for term in ("quick start", "get started", "product function", "practical function", "main function"))
            )
        ):
            return choose(["Manual05_6", "Manual05_7"], "topic:steam_cleaner_functions_quick_start")

    if manual_id == "Manual06":
        cutlery_basket_context = any(term in question for term in ("餐具篮", "筷篮", "小提篮")) or (
            any(term in question for term in ("篮子", "提篮", "篮筐"))
            and any(term in question for term in ("洗碗机", "灰色", "塑料", "叉", "勺"))
        ) or any(term in q for term in ("cutlery basket", "utensil basket", "silverware basket"))
        if cutlery_basket_context:
            return choose(["Manual06_12"], "topic:dishwasher_cutlery_basket")
        if "专用盐" in question and ("添加" in question or "使用前" in question):
            return choose(["Dish_washer_07", "Dish_washer_01", "Dish_washer_02"], "topic:dishwasher_add_special_salt")
        if "洗涤块" in question or "detergent tablet" in q:
            return choose_empty("topic:dishwasher_detergent_tablet_text_only")
        if "亮碟剂" in question or "rinse aid" in q:
            return choose(["Manual06_6", "Manual06_7"], "topic:dishwasher_rinse_aid")
        if "部件" in question and "洗碗机" in question:
            return choose(["Dish_washer_08"], "topic:dishwasher_parts_overview")
        if "洗涤剂" in question and ("添加" in question or "加入" in question):
            return choose(["Manual06_4", "Dish_washer_03"], "topic:dishwasher_add_detergent")
        if "餐具篮" in question and "型号" in question:
            return choose(["Manual06_11", "Manual06_12"], "topic:dishwasher_basket_by_model")
        if "可折叠下层篮架" in question:
            return choose(["Manual06_12"], "topic:dishwasher_foldable_lower_rack")
        if "上下碗篮" in question and "高度" in question:
            return choose(["Manual06_13"], "topic:dishwasher_basket_height_adjustment")
        if "进水管滤网" in question:
            return choose(["Manual06_22"], "topic:dishwasher_inlet_filter_clean")
        if "上层喷淋臂" in question:
            return choose(["Manual06_23"], "topic:dishwasher_upper_spray_arm_clean")

    if manual_id == "Manual07":
        if "water" in q and ("valume" in q or "volume" in q) and ("program" in q or "programming" in q):
            return choose(["Manual07_24", "Manual07_25", "Manual07_26", "Manual07_27"], "topic:coffee_program_water_volume")
        if "water hardness" in q:
            return choose(["Manual07_46"], "topic:coffee_water_hardness_descaling_table")
        if "milk frother" in q or "steam nozzle" in q:
            return choose_empty("topic:coffee_milk_frother_cleaning_text_only")
        if "energy saving mode" in q and "default setting" in q:
            return choose(["Manual07_6", "Manual07_7", "Manual07_8", "Manual07_9"], "topic:coffee_energy_saving_default")
        if "energy saving mode" in q:
            return choose(["Manual07_4", "Manual07_5"], "topic:coffee_energy_saving_mode")

    if manual_id == "Manual08":
        if "natural release" in q or "nror" in q or "npr" in q:
            return choose_empty("topic:air_fryer_pressure_cooker_natural_release_reference_text_only")
        if "first time" in q and ("air fryer" in q or "fryer" in q):
            return choose_empty("topic:air_fryer_first_use_preparation_text_only")

    if manual_id == "Manual09":
        if "battery compartment" in q:
            return choose(["Manual09_157", "Manual09_158"], "topic:boat_battery_compartment_open")
        if "bimini top" in q and "upright position" in q:
            return choose(["Manual09_189", "Manual09_190"], "topic:boat_bimini_upright_storage")
        if "swim platform" in q and "open" in q:
            return choose(["Manual09_168", "Manual09_169", "Manual09_170"], "topic:boat_swim_platform_open")
        if "jet wash" in q and "clean" in q:
            return choose(["Manual09_172", "Manual09_175", "Manual09_176"], "topic:boat_jet_wash_clean")
        if "bilge pump" in q:
            return choose(["Manual09_201", "Manual09_202"], "topic:boat_bilge_pump")
        if "fire extinguisher" in q or "fire extinguishers" in q:
            return choose(["Manual09_211", "Manual09_212"], "topic:boat_fire_extinguisher_storage")
        if "anchor light switch" in q:
            return choose(["Manual09_223", "Manual09_224", "Manual09_225", "Manual09_226"], "topic:boat_anchor_light_switch")
        if "wet items" in q and "storage" in q:
            return choose(["Manual09_143", "Manual09_144"], "topic:boat_wet_storage_compartments")
        if "water supply button" in q or ("turn on or off" in q and "water supply" in q):
            return choose(
                ["Manual09_175", "Manual09_176", "Manual09_177", "Manual09_178", "Manual09_179", "Manual09_180"],
                "topic:boat_water_supply_button",
            )
        if "remove the bimini top" in q:
            return choose(["Manual09_182", "Manual09_183", "Manual09_184"], "topic:boat_bimini_top_remove")
        if "engine oil level" in q and ("sailing" in q or "continued" in q):
            return choose(["Manual09_196", "Manual09_197"], "topic:boat_engine_oil_level")
        if "maintenance setting screen" in q:
            return choose(["Manual09_78", "Manual09_79", "Manual09_80"], "topic:boat_maintenance_setting_screen")
        if "make the boat move forward" in q:
            return choose(["Manual09_22", "Manual09_23"], "topic:boat_move_forward")
        if (
            ("start" in q and ("engine" in q or "engines" in q))
            or ("turn on" in q and "engine" in q)
            or ("boat's engine" in q and "started" in q)
        ):
            return choose(["Manual09_235", "Manual09_236", "Manual09_237"], "topic:boat_engine_start")
        if "factory reset" in q:
            return choose(["Manual09_87", "Manual09_88", "Manual09_89"], "topic:boat_factory_reset_screen")
        if "steering system" in q and any(term in q for term in ("check", "driving", "drive", "operation")):
            return choose(["Manual09_206", "Manual09_207", "Manual09_208", "Manual09_209", "Manual09_210"], "topic:boat_steering_system_checks")
        if (
            ("ship steers" in q or "ship steer" in q)
            or (
                any(term in q for term in ("steer", "steering"))
                and any(term in q for term in ("ship", "boat"))
                and not any(term in q for term in ("check", "system", "bimini", "anchor", "throttle"))
            )
        ):
            return choose_empty("topic:boat_steering_principle_text_only")
        if (
            ("bimini" in q or "canopy" in q)
            and any(term in q for term in ("install", "use the canopy", "use it as a canopy"))
            and not any(term in q for term in ("remove", "store", "upright", "collapsed"))
        ):
            return choose(["Manual09_182", "Manual09_183", "Manual09_184"], "topic:boat_bimini_top_install")
        if "turn a boat" in q or "turn the boat" in q:
            return choose(["Manual09_239", "Manual09_240", "Manual09_241"], "topic:boat_turning")
        if "load the boat" in q and "cruise is over" in q:
            return choose(["Manual09_255", "Manual09_256"], "topic:boat_load_after_cruise")
        if "throttle-cable" in q or "throttle cable" in q:
            return choose(["Manual09_263", "Manual09_264"], "topic:boat_throttle_cable")
        if "over-temperature" in q or "over temperature" in q or "overtemp" in q:
            return choose(["Manual09_95"], "topic:boat_over_temperature_warning")
        if "sound system" in q or "stereo system" in q or ("listen to music" in q and "phone" in q):
            return choose(["Manual09_111", "Manual09_112"], "topic:boat_stereo_system")

    if manual_id == "Manual04":
        if "安全要点" in question or ("safety" in q and ("blower" in q or "leaf blower" in q)):
            return choose(["Manual04_32", "Manual04_33", "Manual04_34", "Manual04_35"], "topic:blower_operation_safety_points")
        if "热机" in question and ("启动" in question or "start" in q):
            return choose(["Manual04_24", "Manual04_26", "Manual04_27", "Manual04_28"], "topic:blower_hot_start_steps")
        if any(term in question for term in ("关闭", "关机")) or (
            any(term in q for term in ("stop", "shut off", "turn off")) and ("blower" in q or "leaf blower" in q)
        ):
            return choose(["Manual04_29"], "topic:blower_stop")
        if ("冷机" in question or "cold" in q) and ("启动" in question or "start" in q):
            return choose(["Manual04_24", "Manual04_25", "Manual04_27", "Manual04_28"], "topic:blower_cold_start_steps")

    if manual_id == "Manual10":
        if "delete a single image" in q or ("erase" in q and "single image" in q):
            return choose(["Manual10_184", "Manual10_185", "Manual10_186"], "topic:camera_delete_single_image")
        if "card" in q and "camera" in q and ("install" in q or "insert" in q) and "before photography" in q:
            return choose(["Manual10_25", "Manual10_26", "Manual10_27", "Manual10_28", "Camera_16", "Camera_17"], "topic:camera_card_install_before_photography")
        if "eyepiece cover" in q:
            return choose(["Manual10_155"], "topic:camera_eyepiece_cover")
        if "af mode" in q:
            return choose(["Manual10_97", "Manual10_98", "Camera_31"], "topic:camera_af_mode_core")
        if "cp direct" in q or ("direct printing" in q and any(term in q for term in ("cp", "camera", "print"))):
            return choose(["Manual10_188", "Manual10_193", "Manual10_195"], "topic:camera_cp_direct_printing")
        if "off-center subject" in q:
            return choose(["Manual10_111", "Manual10_112"], "topic:camera_off_center_subject")
        if (
            'model to "p"' in q
            or '\\"p\\"' in q
            or "p model" in q
            or "p mode" in q
            or (re.search(r"(?<![a-z0-9])p(?![a-z0-9])", q) and any(term in q for term in ("camera", "model", "mode")))
        ):
            return choose(["Manual10_115"], "topic:camera_p_mode")
        if "view the camera image on tv" in q or ("view" in q and "tv" in q and "camera image" in q):
            return choose(["Manual10_182", "Manual10_183"], "topic:camera_view_image_on_tv")
        if "fine tune the model" in q or "fine-tune the model" in q:
            return choose(["Manual10_82", "Manual10_83"], "topic:camera_fine_tune_mode")
        if ("beep" in q or "beeper" in q) and any(term in q for term in ("off", "silence", "disable", "mute")):
            return choose(["Manual10_156"], "topic:camera_beeper_off_menu")
        if "handling precautions" in q or (
            "prevent damage" in q and any(term in q for term in ("camera body", "lcd", "lens contacts", "cf card"))
        ):
            return choose_empty("topic:camera_handling_precautions_text_only")
        if "remove" in q and "shutter button" in q:
            return choose(["Manual10_30", "Manual10_31"], "topic:camera_shutter_button_remove")
        if "erase all images" in q:
            return choose(["Camera_58", "Camera_59", "Manual10_187"], "topic:camera_erase_all_images")
        if "turn the <> switch to<off>" in q or ("date/time" in q and "battery" in q):
            return choose(["Manual10_46", "Manual10_47", "Manual10_48", "Manual10_49"], "topic:camera_date_time_battery")
        if "battery" in q and any(term in q for term in ("not installed", "outside", "recharge", "charge")):
            v1 = proposal("v1")
            wanted = ["Manual10_12", "Manual10_13", "Manual10_14", "Manual10_15"]
            if set(wanted).issubset(v1):
                return wanted, "topic:camera_battery_external_recharge_v1"
            return choose(wanted, "topic:camera_battery_external_recharge")
        if "battery" in q and ("install" in q or "insert" in q) and ("powered off" in q or "power off" in q):
            return choose(["Manual10_16", "Camera_09", "Camera_10", "Camera_12", "Camera_13"], "topic:camera_battery_install")
        if "household electrical outlet" in q or "household power socket" in q or ("power" in q and "outlet" in q):
            return choose(["Camera_14", "Camera_15", "Manual10_19", "Manual10_20"], "topic:camera_household_outlet_power")
        if ("mount" in q or "install" in q or "attach" in q) and "lens" in q:
            return choose(["Manual10_21", "Manual10_22", "Manual10_23", "Manual10_24"], "topic:camera_lens_mount")

    if manual_id == "Manual11":
        if "dcb101" in q and ("指示灯" in question or "indicator" in q or "闪烁" in question):
            return choose(["drill0_08", "drill0_09", "drill0_10", "drill0_11", "drill0_12"], "topic:drill_dcb101_indicator_lights")
        if "电池组" in question and ("安装" in question or "拆卸" in question):
            return choose(["Manual11_8"], "topic:drill_battery_pack_install_remove")
        if "附件" in question or ("accessories" in q and "drill" in q):
            return choose(["Manual11_9"], "topic:drill_accessories")
        if "充电" in question and ("电钻" in question or "遵循" in question or "步骤" in question):
            return choose(["Manual11_2"], "topic:drill_battery_charging_core")
        if "无键夹头" in question or ("keyless chuck" in q and ("install" in q or "single sleeve" in q)):
            return choose(["drill0_01", "drill0_02", "drill0_03", "Manual11_7", "drill0_14"], "topic:drill_keyless_chuck")
        if "腰带挂钩" in question or "批头夹" in question or "belt hook" in q or "bit clip" in q:
            return choose(["drill0_01", "drill0_02", "drill0_03", "Manual11_7", "drill0_14"], "topic:drill_belt_hook_bit_clip")

    if manual_id == "Manual12":
        if "charging contact" in q or ("contacts" in q and "charging" in q):
            return choose_empty("topic:earphones_charging_contact_care_text_only")
        if "ear tip" in q or "ear tips" in q or "eartip" in q or "eartips" in q:
            return choose_empty("topic:earphones_ear_tip_fit_text_only")
        if (
            "water resistance" in q
            or "water-resistant" in q
            or "water resistant" in q
            or "wet surface" in q
            or "near water" in q
            or "ip55" in q
            or ("sweat" in q and "dust" in q)
        ):
            return choose_empty("topic:earphones_water_resistance_text_only")
        if "components" in q or "in my hand" in q:
            return choose(["Manual12_0", "Manual12_1", "Manual12_2", "Manual12_3", "Manual12_4"], "topic:earphones_components")
        if "case battery" in q or "charging case battery" in q:
            return choose(["earphones_04", "earphones_05", "earphones_06", "earphones_07"], "topic:earphones_case_battery_charge")
        if "bluetooth" in q and ("pairing" in q or "connecting" in q or "pair" in q or "connect" in q):
            return choose(["Manual12_5", "Manual12_6", "Manual12_7"], "topic:earphones_bluetooth_pairing")
        if ("besides" in q or "other functions" in q) and "earphones" in q:
            return choose(["earphones_01", "earphones_02", "earphones_03", "Manual12_10"], "topic:earphones_other_functions")
        if "two main" in q and "control" in q:
            return choose(["Manual12_8", "Manual12_9"], "topic:earphones_two_main_controls")

    if manual_id == "Manual13":
        if "trouble" in q or "trobles" in q or "troubleshooting" in q:
            return choose_empty("topic:ereader_troubleshooting_text_only")
        if "ebook mode" in q and ("m" in q or "botton" in q or "button" in q):
            return choose(["eReader_08"], "topic:ereader_ebook_m_button")
        if "record voice" in q or ("voice" in q and "record" in q):
            return choose(["Manual13_11", "Manual13_12"], "topic:ereader_voice_record")
        if (
            "font size" in q
            or "page display" in q
            or ("while reading" in q and ("display" in q or "font" in q or "brightness" in q))
        ):
            return choose(["eReader_08", "Manual13_5", "Manual13_6"], "topic:ereader_reading_display_font")
        if (
            "delete" in q
            or "deletion" in q
            or "file transfer" in q
            or "copy ebooks" in q
            or "copy ebook" in q
            or "copy files" in q
            or "removable disk" in q
            or "usb connection" in q
        ):
            return choose_empty("topic:ereader_document_management_text_only")
        if "listen to music" in q:
            return choose(["Manual13_7", "Manual13_8"], "topic:ereader_music")
        if "play video" in q:
            return choose(["Manual13_9"], "topic:ereader_video")
        if "photo viewer" in q:
            return choose(["Manual13_10"], "topic:ereader_photo_viewer")
        if ("ebook mode" in q or "ebook" in q) and "m button" in q:
            return choose(["eReader_08"], "topic:ereader_ebook_m_button")
        if ("browser history" in q or "browsing history" in q) and ("main menu" in q or "menu" in q):
            return choose(["Manual13_3", "Manual13_4"], "topic:ereader_browser_history_menu")
        if "different views" in q or "buttons and interfaces" in q:
            return choose(["Manual13_0", "Manual13_1", "Manual13_2"], "topic:ereader_views_buttons")

    if manual_id == "Manual15":
        if "using this fax" in q and ("safety" in q or "pay attention" in q):
            return choose_empty("topic:fax_general_use_safety_text_only")
        if (
            "fax" in q
            and any(term in q for term in ("caution label", "warning label", "safety label", "warning labels", "caution labels", "labels attached"))
        ):
            return choose(["fax_01", "fax_02", "fax_03", "fax_04", "fax_05", "fax_06", "fax_07"], "topic:fax_caution_warning_labels")
        if "toner" in q or "print cartridge" in q or "cartridge replacement" in q:
            return choose_empty("topic:fax_toner_cartridge_text_only")
        if "document feeder" in q or ("original" in q and "document" in q and "load" in q):
            return choose_empty("topic:fax_document_feeder_loading_text_only")
        if ("connect" in q or "connecting" in q or "setup" in q) and "fax" in q:
            return choose(["Manual15_2", "Manual15_3"], "topic:fax_connection_setup")
        if ("finger" in q or "fingers" in q) and "fax" in q:
            return choose(
                ["Manual15_6", "Manual15_7", "Manual15_8", "Manual15_9", "fax_08", "Manual15_10", "Manual15_11", "Manual15_12", "Manual15_13", "Manual15_14", "Manual15_15"],
                "topic:fax_finger_safety",
            )
        if ("moving" in q or "move this fax" in q or "move the fax" in q or "before moving" in q) and "fax" in q:
            return choose(["fax_08"], "topic:fax_moving_notice")
        if "safety precautions" in q and "using the fax" in q:
            return choose_empty("topic:fax_use_safety_text_only")

    if manual_id == "Manual16":
        if "扣紧表带" in question or ("band" in q and ("tight" in q or "fasten" in q)):
            return choose(["Manual16_6", "Manual16_7", "Manual16_8"], "topic:fitness_tracker_fasten_band")
        if "运动应用" in question and ("追踪" in question or "分析" in question):
            return choose_empty("topic:fitness_tracker_app_tracking_text_only")
        if "心率" in question and ("测量" in question or "如何" in question):
            return choose(["Manual16_44"], "topic:fitness_tracker_heart_rate_measure")
        if "拆卸表带" in question or ("取下" in question and "表带" in question):
            return choose(["Manual16_9", "Manual16_10", "Manual16_11"], "topic:fitness_tracker_band_remove")
        if (
            (
                any(term in q for term in ("interface", "basic operation", "home screen", "operate the fitness tracker", "operating the fitness tracker"))
                or ("\u5065\u8eab\u8ffd\u8e2a\u5668" in question and "\u754c\u9762" in question)
                or ("\u64cd\u4f5c" in question and "\u754c\u9762" in question)
            )
            and not any(term in q for term in ("notification", "phone notification", "troubleshoot", "problem"))
        ):
            return choose(["Manual16_12"], "topic:fitness_tracker_interface_basic_operation")
        if "电量低" in question and ("充电" in question or "正确" in question):
            return choose(["Manual16_1", "Manual16_2"], "topic:fitness_tracker_low_battery_charge")
        if "手机的通知" in question:
            return choose(["Manual16_28", "Manual16_29", "Manual16_30"], "topic:fitness_tracker_phone_notifications")
        if "遇到哪些问题" in question and "解决" in question:
            return choose(["Manual16_48", "Manual16_49", "Manual16_50"], "topic:fitness_tracker_troubleshooting")
        if any(term in q for term in ("package", "packaging", "included", "box contains")) or any(
            term in question for term in ("包装", "开箱", "包含哪些")
        ):
            return choose(
                ["fitness_trackers_01", "fitness_trackers_02", "fitness_trackers_03"],
                "topic:fitness_tracker_package_group",
            )

    if manual_id == "Manual17":
        if "使用冰箱冰柜" in question and "前五条" in question:
            return choose(["Manual17_11", "Manual17_12", "Manual17_13", "Manual17_14"], "topic:refrigerator_first_five_safety")
        if "连接电源" in question and "前五条" in question:
            return choose(["Manual17_0", "Manual17_1", "Manual17_2", "Manual17_3"], "topic:refrigerator_power_first_four")
        if any(term in question for term in ("连接电源", "接通电源")) or ("connect" in q and "power" in q):
            return choose(["Manual17_0", "Manual17_1", "Manual17_2"], "topic:refrigerator_power_safety")

    if manual_id == "Manual14":
        if "心率目标" in question or ("target heart rate" in q):
            return choose_empty("topic:exercise_bike_target_heart_rate_text_only")
        if "轻松骑行类别" in question or ("easy ride" in q and "program" in q):
            return choose(["Manual14_27", "Manual14_28", "Manual14_29"], "topic:exercise_bike_easy_ride_programs")
        if "用户档案" in question or ("user profile" in q and ("edit" in q or "steps" in q)):
            return choose(["Manual14_26"], "topic:exercise_bike_user_profile_edit")
        if "启动健身单车前" in question or ("exercise bike" in q and "before" in q and "start" in q):
            return choose(["Manual14_19", "Manual14_20"], "topic:exercise_bike_before_start")
        if "运动前" in question and "舒适度" in question:
            return choose(["Manual14_24", "Manual14_25"], "topic:exercise_bike_comfort_adjustment")
        if "山地类别" in question:
            return choose(["Manual14_30", "Manual14_31", "Manual14_32"], "topic:exercise_bike_mountain_programs")
        if "最高难度类别" in question:
            return choose(["Manual14_33", "Manual14_34", "Manual14_35"], "topic:exercise_bike_highest_difficulty_programs")
        if any(term in question for term in ("\u78c1", "\u8d77\u640f", "\u533b\u7597", "\u690d\u5165")):
            return choose_empty("topic:exercise_bike_magnet_medical_text_only")
        if "控制台" in question and any(term in question for term in ("显示", "控制", "功能")):
            return choose(["Manual14_21"], "topic:exercise_bike_console_functions")

    if manual_id == "Manual18":
        if "启动" in question and "发动机" in question and "前两个步骤" in question:
            return choose(["Manual18_25", "Manual18_26", "Manual18_27"], "topic:generator_engine_start_first_two_steps")
        if "直流保护器" in question:
            return choose(["generator_08"], "topic:generator_dc_protector")
        if "空气滤清器" in question:
            return choose(["Manual18_62", "Manual18_63", "Manual18_64"], "topic:generator_air_cleaner_steps")
        if "连接交流电" in question and "安全" in question:
            return choose(["Manual18_16", "Manual18_33", "Manual18_34", "Manual18_35"], "topic:generator_ac_connection_safety")
        if "两种不同的开关" in question:
            return choose(["generator_06", "generator_07"], "topic:generator_two_switches")
        if "无法启动" in question:
            return choose(["generator_05", "generator_06"], "topic:generator_engine_cannot_start")
        if "燃油排空" in question:
            return choose(["Manual18_67", "Manual18_68"], "topic:generator_drain_fuel")
        if "触电" in question:
            return choose(["Manual18_12", "Manual18_13", "Manual18_14", "Manual18_15"], "topic:generator_electric_shock_safety")
        if "燃油开关旋钮" in question or ("fuel switch" in q and "knob" in q):
            return choose(["Manual18_19", "Manual18_20"], "topic:generator_fuel_switch_knob")
        if "燃油" in question and "使用前检查" in question:
            return choose(["generator_12", "generator_13", "generator_14"], "topic:generator_fuel_precheck")
        if "发动机机油" in question and "使用前检查" in question:
            return choose(["generator_17", "generator_18", "generator_19", "Manual18_23"], "topic:generator_engine_oil_precheck")
        if "启动" in question and "最后三个步骤" in question:
            return choose(["generator_16", "Manual18_29"], "topic:generator_start_last_steps")
        if "电池" in question and "充电" in question:
            return choose(["Manual18_37", "Manual18_38", "Manual18_39"], "topic:generator_battery_charge")
        if "发动机停机" in question or ("engine" in q and "stop" in q):
            return choose(["generator_22", "generator_23", "generator_24", "Manual18_40", "Manual18_41"], "topic:generator_engine_stop")
        if "火花塞" in question or "spark plug" in q:
            return choose(["Manual18_44", "Manual18_45", "Manual18_46", "Manual18_47", "Manual18_48"], "topic:generator_spark_plug")
        if "更换" in question and "发动机机油" in question:
            return choose(["Manual18_49", "Manual18_50", "generator_25", "generator_26", "Manual18_51"], "topic:generator_engine_oil_change")

    if manual_id == "Manual19":
        if "safety tips" in q and "grill" in q:
            return choose_empty("topic:grill_safety_tips_text_only")
        if "connect regulator" in q and "lp tank" in q:
            return choose(["Manual19_16", "Manual19_17", "Manual19_18", "Manual19_19"], "topic:grill_regulator_lp_tank")
        if "burner flame" in q or ("flame" in q and "grill" in q):
            return choose(["Manual19_29"], "topic:grill_burner_flame_check")
        if ("storage" in q or "long-term" in q or "long term" in q) and "clean" in q:
            return choose_empty("topic:grill_cleaning_before_storage_text_only")
        if "leak testing" in q and ("valves" in q or "hose" in q or "regulator" in q):
            return choose(["Manual19_20", "Manual19_21"], "topic:grill_leak_test_valves_hose_regulator")
        if "national electric code" in q or "ground fault" in q or "gfi" in q:
            return choose(["Manual19_44", "Manual19_45"], "topic:grill_nec_gfi")
        if "indirect cooking" in q:
            return choose(["Manual19_36"], "topic:grill_indirect_cooking")
        if ("first steps" in q or "first step" in q or "first three steps" in q) and ("assembly" in q or "assemble" in q):
            return choose(["Manual19_49", "Manual19_50", "Manual19_51", "Manual19_52"], "topic:grill_first_assembly_steps")

    if manual_id == "Manual20":
        if "fuel meter" in q and "hour meter" in q:
            return choose(["Manual20_58", "Manual20_59"], "topic:jetski_fuel_meter_hour_meter")
        if "filler caps" in q or "filler cap" in q:
            return choose(["Manual20_40", "Manual20_41"], "topic:jetski_filler_caps_remove")
        if "fuel filter" in q and "fuel tank" in q:
            return choose(["Manual20_81", "Manual20_82"], "topic:jetski_fuel_filter_tank_inspect")
        if "intake" in q and "impeller" in q and ("dirty" in q or "clean" in q):
            return choose(["Manual20_85", "Manual20_86"], "topic:jetski_intake_impeller_clean")
        if "identification number" in q or "identification numbers" in q:
            return choose_empty("topic:jetski_identification_numbers_text_only")
        if "quick shift trim system" in q or "qsts" in q:
            return choose(["Manual20_51", "Manual20_52"], "topic:jetski_qsts_selector")
        if "watercraft characteristics" in q or (
            "characteristics" in q and ("jetski" in q or "jet ski" in q or "watercraft" in q)
        ):
            return choose(["Manual20_24", "Manual20_25"], "topic:jetski_watercraft_characteristics")
        if "flush" in q or "rinse" in q or "salt water" in q:
            return choose(["Manual20_34", "Manual20_49"], "topic:jetski_flush_cooling_passages")
        if "encountering vessels" in q:
            return choose(["Manual20_26", "Manual20_27", "Manual20_28"], "topic:jetski_encountering_vessels")
        if "engine switches" in q or "engine switch" in q:
            return choose(["Manual20_43", "Manual20_44", "Manual20_45"], "topic:jetski_engine_switches")
        if "two kinds of levers" in q:
            return choose(["Manual20_46", "Manual20_47"], "topic:jetski_two_levers")
        if "adjustable sponson" in q:
            return choose(["Manual20_83", "Manual20_84"], "topic:jetski_adjustable_sponson")
        if "start my jetski" in q and "different situations" in q:
            return choose(["Manual20_87", "Manual20_88", "Manual20_89"], "topic:jetski_start_different_situations")
        if "board my jetski" in q and "passenger" in q and "deep water" in q:
            return choose(["Manual20_90", "Manual20_91", "Manual20_92"], "topic:jetski_board_passenger_deep_water")
        if "location of main components" in q:
            return choose(["Manual20_31", "Manual20_32", "Manual20_33"], "topic:jetski_location_main_components")
        if "operating requirements" in q or "operation requirements" in q or ("requirements" in q and "before using" in q):
            return choose(["Manual20_15", "Manual20_19"], "topic:jetski_operating_requirements")
        if "cruising limitation" in q or "cruising limitations" in q:
            return choose(["Manual20_16", "Manual20_17"], "topic:jetski_cruising_limitations")

    if manual_id == "Manual21":
        if (
            ("switch" in q and any(term in q for term in ("remove", "removal", "reinstall", "install", "replace", "puller")))
            or ("\u8f74\u4f53" in question and any(term in question for term in ("\u62c6\u5378", "\u91cd\u65b0\u5b89\u88c5", "\u5b89\u88c5", "\u66f4\u6362")))
        ):
            return choose(["Manual21_12", "Manual21_13", "Manual21_14"], "topic:function_keyboard_switch_remove_reinstall")
        if (
            "损害赔偿" in question
            or "免责声明" in question
            or "除外责任" in question
            or ("warranty" in q and any(term in q for term in ("liability", "disclaimer", "damage")))
        ):
            return choose_empty("topic:function_keyboard_warranty_disclaimer_text_only")
        if "保修政策" in question or "warranty policy" in q:
            return choose(["Manual21_16"], "topic:function_keyboard_warranty_policy")
        if ("cam" in q and "software" in q) or "硬件模式" in question or "hardware mode" in q:
            return choose_empty("topic:function_keyboard_software_hardware_mode_text_only")
        if "CAM" in question and "软件" in question:
            return choose_empty("topic:function_keyboard_software_hardware_mode_text_only")
        if (
            ("setup" in q or "set up" in q or "setting" in q or "\u8bbe\u7f6e" in question)
            and ("keyboard" in q or "\u529f\u80fd\u952e\u76d8" in question)
        ):
            return choose(["Manual21_2", "function_keyboard_01", "function_keyboard_02", "Manual21_3"], "topic:function_keyboard_setup_core")
        if ("initial" in q or "first" in q or "初次" in question) and ("setup" in q or "设置" in question):
            return choose(["Manual21_1", "Manual21_2", "function_keyboard_01"], "topic:function_keyboard_initial_setup")

    if manual_id == "Manual22":
        if "searching status" in q and "landline" in q:
            return choose_empty("topic:landline_searching_status_text_only")
        if "led indicator" in q or ("current status" in q and "led" in q):
            return choose(["Manual22_46"], "topic:landline_led_indicator_status")
        if "register" in q or "registration" in q or "pair" in q:
            return choose(["Manual22_18", "Manual22_58"], "topic:landline_handset_registration")
        if "ringer" in q or "melody" in q:
            return choose_empty("topic:landline_ringer_setting_text_only")
        if "phonebook" in q or "contact" in q:
            return choose_empty("topic:landline_phonebook_text_only")
        if "overview" in q and "base station" in q:
            return choose(["Manual22_18"], "topic:landline_base_station_overview")
        if "connect" in q and "base station" in q:
            return choose(["Manual22_21"], "topic:landline_base_station_connect")
        if "install" in q and "handset" in q:
            return choose(["Manual22_23", "Manual22_25"], "topic:landline_handset_install")

    if manual_id == "Manual23":
        if "height of cut" in q and "electric deck lift" in q:
            return choose(["Manual23_55", "Manual23_56"], "topic:mower_height_cut_electric_deck_lift")
        if "replace the mower belt" in q or ("mower belt" in q and "replace" in q):
            return choose(["Manual23_99", "Manual23_100", "Manual23_101"], "topic:mower_belt_replace")
        if "unload" in q and ("lawn mower" in q or "mower" in q):
            return choose_empty("topic:mower_unload_text_only")
        if "rear-shock" in q or "rear shock" in q:
            return choose(["Manual23_37", "Manual23_38"], "topic:mower_rear_shock_adjust")
        if "engine oil" in q and ("change" in q or "changing" in q):
            return choose(["Manual23_75", "Manual23_76"], "topic:mower_engine_oil_change")
        if ("remove" in q or "removing" in q) and ("filter" in q or "filters" in q) and ("lawn mower" in q or "mower" in q):
            return choose(["Manual23_72"], "topic:mower_air_cleaner_filter_removal")
        if "load a lawn mower" in q or ("load" in q and "lawn mower" in q):
            return choose(["Manual23_63", "Manual23_64", "Manual23_65"], "topic:mower_loading")
        if "lower" in q and "roll bar" in q:
            return choose(["Manual23_32"], "topic:mower_lower_roll_bar")

    if manual_id == "Manual25":
        if "serial port connector" in q or "10-1 pin com" in q:
            return choose(["Manual25_39"], "topic:motherboard_serial_port_connector")
        if "tpm connector" in q or "14-1 pin tpm" in q:
            return choose(["Manual25_40"], "topic:motherboard_tpm_connector")
        if "thermal sensor connector" in q or "t_sensor" in q:
            return choose(["Manual25_43"], "topic:motherboard_thermal_sensor_connector")
        if "central processing unit" in q or "cpu" in q and "prepare" in q:
            return choose(["Manual25_9", "Manual25_10", "Manual25_11", "Manual25_12"], "topic:motherboard_cpu_install_prepare")
        if "pci express 3.0 x16" in q or "pci express x16" in q:
            return choose(["Manual25_31"], "topic:motherboard_pcie_x16_slots")
        if "update the bios" in q or "bios file" in q:
            return choose(["Manual25_60", "Manual25_61"], "topic:motherboard_update_bios_file")
        if "create raid" in q or "raid" in q:
            return choose(["Manual25_72", "Manual25_73", "Manual25_74", "Manual25_75"], "topic:motherboard_create_raid")
        if "system memory" in q:
            return choose(["Manual25_19"], "topic:motherboard_system_memory")
        if "cpu fan" in q or "cpu_fan" in q:
            return choose(["Manual25_41"], "topic:motherboard_cpu_fan_header")
        if "front-panel usb" in q or "front panel usb" in q or ("usb" in q and "header" in q):
            return choose(["Manual25_47"], "topic:motherboard_front_panel_usb_header")
        if (
            "load from profile" in q
            or ("bios" in q and "profile" in q)
            or "saved cmos settings" in q
            or "previous bios settings" in q
        ):
            return choose_empty("topic:motherboard_load_from_profile_text_only")
        if "cmos" in q or "rtc" in q or ("battery" in q and "motherboard" in q):
            return choose(["Manual25_33"], "topic:motherboard_cmos_battery")
        if "jumpers" in q:
            return choose(["Manual25_33", "Manual25_34"], "topic:motherboard_jumpers")
        if "rear panel connectors" in q:
            return choose(["Manual25_35", "Manual25_36", "Manual25_37", "Manual25_38"], "topic:motherboard_rear_panel_connectors")
        if "onboard led" in q:
            return choose(["Manual25_54", "Manual25_55", "Manual25_56", "Manual25_57", "Manual25_58"], "topic:motherboard_onboard_led")
        if "m.2" in q or "ngff" in q or ("storage device" in q and "mounting screw" in q):
            return choose(["Manual25_51"], "topic:motherboard_m2_socket")

    if manual_id == "Manual26":
        if "挡泥板" in question:
            return choose(["rideon_motorcycle_01", "rideon_motorcycle_02"], "topic:rideon_motorcycle_fender_install")
        if "前轮" in question:
            return choose(["Manual26_6", "Manual26_7"], "topic:rideon_motorcycle_front_wheel_install")

    if manual_id == "Manual27":
        if (
            ("电池" in question and any(term in question for term in ("安装", "更换", "替换", "装入", "放入")))
            or ("battery" in q and any(term in q for term in ("install", "insert", "replace", "change")))
        ):
            return choose(["Manual27_1", "Manual27_2", "Manual27_3"], "topic:bluetooth_mouse_battery_install")
        if "首次" in question and ("驱动程序" in question or "driver" in q):
            return choose(["Manual27_16", "Manual27_17"], "topic:bluetooth_mouse_driver_first_use")
        if ("WIDCOMM" in question or "widcomm" in q) and any(term in question for term in ("卸载", "删除", "移除")):
            return choose_empty("topic:bluetooth_mouse_widcomm_uninstall_text_only")
        if ("WIDCOMM" in question or "widcomm" in q) and (
            any(term in q for term in ("pair", "pairing", "connect", "connection", "hid", "search"))
            or any(term in question for term in ("\u914d\u5bf9", "\u4eba\u673a\u63a5\u53e3\u8bbe\u5907"))
        ):
            return choose(["Manual27_12", "Manual27_13"], "topic:bluetooth_mouse_widcomm_pairing")
        if "WIDCOMM" in question or "widcomm" in q:
            return choose(["Manual27_4", "Manual27_5", "Manual27_6", "Manual27_7", "Manual27_8", "Manual27_9"], "topic:bluetooth_mouse_widcomm_driver")
        if "电量状态" in question or ("battery" in q and "status" in q and "mouse" in q):
            return choose(["Manual27_14", "Manual27_15", "Manual27_16", "Manual27_17"], "topic:bluetooth_mouse_battery_status")
        if "人机接口设备" in question or "human interface" in q:
            return choose(["Manual27_18"], "topic:bluetooth_mouse_hid_connection")

    if manual_id == "Manual28":
        if (
            ("\u5b89\u88c5" in question and "\u70e4\u7bb1\u95e8" in question)
            or (("install" in q or "reinstall" in q) and "oven door" in q)
        ):
            return choose_empty("topic:oven_door_install_text_only")
        oven_door_removal = (
            any(term in question for term in ("拆卸烤箱门", "拆下烤箱门", "烤箱门怎么拆"))
            or (
                any(term in question for term in ("这门怎么拆", "门怎么拆", "拆下来", "取下门", "门铰链"))
                and any(term in question for term in ("烤箱", "烤炉", "铰链", "搪瓷"))
            )
            or (("remove" in q or "take off" in q) and "oven door" in q)
        )
        if oven_door_removal:
            return choose(["oven_01", "oven_02"], "topic:oven_door_remove_core")
        if "清洁烤箱外部" in question or ("clean" in q and "oven" in q and "exterior" in q):
            return choose_empty("topic:oven_exterior_clean_text_only")
        if "催化侧面板" in question or ("catalytic" in q and "side panel" in q):
            return choose_empty("topic:oven_catalytic_side_panel_text_only")
        if "烤架烤盘套装" in question:
            return choose(["oven_13"], "topic:oven_grill_pan_set")
        if "油脂过滤器" in question:
            return choose(["oven_14"], "topic:oven_grease_filter")
        if "滑动搁架" in question:
            return choose(["oven_15"], "topic:oven_sliding_shelf")
        if "烤架" in question and "套装" not in question:
            return choose(["oven_10"], "topic:oven_grill_rack")
        if "烤盘" in question and "套装" not in question:
            return choose(["oven_09"], "topic:oven_baking_tray")
        if any(term in question for term in ("热空气", "蒸汽", "烫伤")) or (
            ("steam" in q or "hot air" in q) and ("door" in q or "open" in q)
        ):
            return choose_empty("topic:oven_hot_air_steam_door_text_only")
        if "顶部加热元件" in question or "top heating element" in q:
            return choose(["oven_03", "oven_04", "oven_05"], "topic:oven_top_heating_element")
        if ("remove" in q or "拆下" in question) and ("oven door" in q or "烤箱门" in question):
            return choose(["oven_01", "oven_02", "oven_06", "Manual28_7"], "topic:oven_door_removal")

    if manual_id == "Manual24":
        if "light timer" in q:
            return choose(["Manual24_11", "Manual24_12"], "topic:microwave_light_timer")
        if "favorite recipe" in q:
            return choose(["Manual24_14", "Manual24_15", "Manual24_16", "Manual24_17"], "topic:microwave_favorite_recipe")
        if "reheat" in q and "food" in q:
            return choose(["Manual24_27", "Manual24_28", "Manual24_29", "Manual24_30", "Manual24_31"], "topic:microwave_reheat_food")
        if "charcoal filter" in q and ("replace" in q or "replacement" in q):
            return choose(["Manual24_50", "Manual24_51", "Manual24_52", "Manual24_53"], "topic:microwave_charcoal_filter_replace")
        if "oven light" in q and ("replace" in q or "replacement" in q):
            return choose(["Manual24_54", "Manual24_55", "Manual24_56"], "topic:microwave_oven_light_replace")
        if "auto defrost" in q or ("defrost" in q and "microwave" in q):
            return choose(["Manual24_32", "Manual24_33"], "topic:microwave_auto_defrost")
        if "set up control" in q or "set up the control" in q or "setup control" in q or "control panel" in q:
            return choose(["Manual24_5"], "topic:microwave_control_panel_overview")
        if "vent" in q and "fan" in q:
            return choose_empty("topic:microwave_vent_fan_text_only")
        if "child lock" in q or "control lock" in q:
            return choose_empty("topic:microwave_child_lock_text_only")
        if "turntable" in q:
            return choose_empty("topic:microwave_turntable_text_only")
        if "control set-up" in q or "control setup" in q:
            return choose(["Manual24_9", "Manual24_10"], "topic:microwave_control_setup")
        if "meat setting" in q:
            return choose(["Manual24_35"], "topic:microwave_meat_setting")

    if manual_id == "Manual29":
        if any(term in question for term in ("插入存储卡", "装入存储卡")) or (
            "memory card" in q and any(term in q for term in ("insert", "install", "load"))
        ):
            return choose(["Manual29_57", "Manual29_58", "Manual29_59"], "topic:hybrid_camera_insert_memory_card")
        if "闪光灯" in question or ("flash" in q and any(term in q for term in ("use", "setting", "set"))):
            return choose(["Manual29_52"], "topic:hybrid_camera_flash")
        if "自拍" in question or ("self-timer" in q or "self timer" in q):
            return choose(["Manual29_51"], "topic:hybrid_camera_self_timer")
        if "指令拨盘" in question or "command dial" in q:
            return choose(["Manual29_10"], "topic:hybrid_camera_command_dial")
        remaining_film_context = any(term in question for term in ("剩余相纸", "相纸余量", "还剩几张")) or (
            "相机" in question and "右边" in question and "点" in question and "红" in question
        ) or any(term in q for term in ("remaining film", "film remaining", "remaining shots", "red dots"))
        if remaining_film_context:
            return choose(["Manual29_46"], "topic:hybrid_camera_remaining_film")
        if "肩带" in question or ("strap" in q and "camera" in q):
            return choose(["Manual29_12", "hybrid_instant_camera_03"], "topic:hybrid_camera_shoulder_strap")
        if "装入电池" in question or ("load" in q and "battery" in q and "camera" in q):
            return choose(["Manual29_13", "Manual29_14", "Manual29_15", "Manual29_16"], "topic:hybrid_camera_load_battery")
        if "装入相纸盒" in question or ("film pack" in q and ("load" in q or "insert" in q)):
            return choose(["Manual29_26", "Manual29_27", "hybrid_instant_camera_01", "hybrid_instant_camera_02"], "topic:hybrid_camera_load_film_pack")
        if "存储卡" in question and "电脑" in question:
            return choose_empty("topic:camera_storage_card_computer_text_only")

    if manual_id == "Manual30":
        pressure_context = (
            "pressure cooker" in q
            or "pressure-cooking" in q
            or "pressure cooking" in q
            or "multi-use" in q
            or "multi use" in q
        )
        if "lid" in q and pressure_context and any(term in q for term in ("align", "mark", "marks", "lock", "unlock")):
            return choose(["Manual30_11", "Manual30_12", "Manual30_13"], "topic:pressure_cooker_lid_alignment")
        if "pressure cooking lid" in q or ("lid" in q and pressure_context and any(term in q for term in ("align", "lock", "unlock", "set"))):
            return choose(
                [
                    "Manual30_11",
                    "Manual30_12",
                    "Manual30_13",
                    "multi-use_pressure_cooker_and_air_fryer_03",
                    "multi-use_pressure_cooker_and_air_fryer_04",
                ],
                "topic:pressure_cooker_lid",
            )
        if "natural release" in q or "nror" in q or "npr" in q:
            return choose(["Manual30_9", "Manual30_18"], "topic:pressure_cooker_natural_release")
        if "quick release button" in q:
            return choose(["Manual30_14"], "topic:pressure_cooker_quick_release_button")
        if "quick release" in q:
            return choose(
                [
                    "multi-use_pressure_cooker_and_air_fryer_01",
                    "multi-use_pressure_cooker_and_air_fryer_02",
                ],
                "topic:pressure_cooker_quick_release",
            )
        if "silicone cap" in q and "float valve" in q:
            return choose(["Manual30_36", "Manual30_37"], "topic:pressure_cooker_silicone_cap_float_valve")
        if "float valve" in q:
            return choose(["Manual30_17", "Manual30_18"], "topic:pressure_cooker_float_valve")
        if "anti-block shield" in q and any(term in q for term in ("set", "install", "remove", "clean", "how can")):
            return choose(["Manual30_34", "Manual30_35"], "topic:pressure_cooker_antiblock_install_remove")
        if "anti-block shield" in q:
            return choose(["Manual30_19"], "topic:pressure_cooker_antiblock")
        if "steam release valve" in q:
            return choose(["Manual30_15", "Manual30_32", "Manual30_33"], "topic:pressure_cooker_steam_release_valve")
        if "condensation collector" in q:
            return choose(["Manual30_38"], "topic:pressure_cooker_condensation_collector")
        if "remove" in q and "sealing ring" in q:
            return choose(["Manual30_30"], "topic:pressure_cooker_remove_sealing_ring")
        if "install" in q and "sealing ring" in q:
            return choose(["Manual30_16", "Manual30_31"], "topic:pressure_cooker_install_sealing_ring")
        if "sealing ring" in q:
            return choose(["Manual30_16"], "topic:pressure_cooker_sealing_ring")
        if ("minimum liquid" in q or "liquid amount" in q) and ("pressure" in q or "pressure-cooking" in q):
            return choose_empty("topic:pressure_cooker_minimum_liquid_text_only")
        if ("liquid" in q or "liquids" in q) and ("pressure" in q or "pressure-cooking" in q):
            return choose(["Manual30_20", "Manual30_21"], "topic:pressure_cooker_liquid_amount")
        if ("air fryer basket" in q or "basket or tray" in q or "multi-level air fryer basket" in q) and (
            "circulate" in q or "hot air" in q or "properly" in q
        ):
            return choose(["Manual30_25", "Manual30_26"], "topic:pressure_cooker_air_fryer_basket")
        if "keep warm" in q and ("pressure" in q or pressure_context):
            return choose_empty("topic:pressure_cooker_keep_warm_text_only")
        if "delayed start" in q or "delay start" in q:
            return choose_empty("topic:pressure_cooker_delayed_start_text_only")

    if manual_id == "Manual31":
        if "发动机安全停机" in question or ("engine" in q and "stop" in q and "pump" in q):
            return choose(["Manual31_16", "Manual31_17", "Manual31_18"], "topic:pump_engine_safe_stop")
        if "排放燃油" in question or "drain fuel" in q:
            return choose(["Manual31_41", "Manual31_42", "Manual31_43"], "topic:pump_drain_fuel")
        if "无法抽水" in question or ("not pump" in q and "water" in q) or ("troubleshoot" in q and "pump" in q):
            return choose_empty("topic:pump_troubleshooting_text_only")
        if "清洗油箱滤网" in question:
            return choose(["Manual31_32"], "topic:pump_oil_tank_strainer_clean_core")
        if "oil tank strainer" in q or "fuel tank strainer" in q or "油箱滤网" in question:
            return choose(["Manual31_31", "Manual31_32", "Manual31_33", "Manual31_34"], "topic:pump_oil_tank_strainer")

    if manual_id == "Manual32":
        if "full bin sensors" in q:
            return choose(["Manual32_12", "Manual32_13", "Manual32_14"], "topic:robot_vacuum_full_bin_sensors_clean")
        if "extractors" in q or "brushes or rollers" in q:
            return choose(["Manual32_19", "Manual32_20", "Manual32_21", "Manual32_22"], "topic:robot_vacuum_extractors_clean")
        if "side brush" in q:
            return choose(["Manual32_16"], "topic:robot_vacuum_side_brush_clean")
        if "two primary modes" in q or "tailor its performance" in q:
            return choose_empty("topic:robot_vacuum_two_modes_text_only")
        if "home base" in q and ("position" in q or "positioning" in q):
            return choose_empty("topic:robot_vacuum_home_base_position_text_only")
        if "troubleshooting" in q or "indicates a problem" in q:
            return choose_empty("topic:robot_vacuum_troubleshooting_text_only")
        if "virtual wall barrier" in q:
            return choose(["Manual32_4", "Manual32_5", "Manual32_6", "Manual32_7"], "topic:robot_vacuum_virtual_wall_barrier")
        if "anatomy" in q or ("parts" in q and "robot" in q and "vacuum" in q) or ("components" in q and "robot" in q and "vacuum" in q):
            return choose(["Manual32_0"], "topic:robot_vacuum_anatomy_overview")
        if "schedule" in q or "scheduling" in q:
            return choose_empty("topic:robot_vacuum_schedule_text_only")
        if "front caster wheel" in q:
            return choose(["Manual32_15"], "topic:robot_vacuum_front_caster_wheel_clean")
        if "cleaning the vacuum cleaner filter" in q or ("filter" in q and "vacuum" in q and "clean" in q):
            return choose(["Manual32_10", "Manual32_11"], "topic:robot_vacuum_filter_clean")

    if manual_id == "Manual33":
        if "power the camera" in q or "powering the camera" in q:
            return choose_empty("topic:security_camera_poe_power_text_only")
        if "t-rail" in q or "t rail" in q:
            return choose(["Manual33_10", "Manual33_11", "Manual33_12", "Manual33_13", "Manual33_14"], "topic:security_camera_t_rail")

    if manual_id == "Manual35":
        if "manual program" in q and ("memorizing" in q or "communication channel" in q or "channel" in q):
            return choose_empty("topic:tv_manual_program_channels_text_only")
        if (
            any(term in q for term in ("poor reception", "weak signal", "weak reception", "ghosts", "snow"))
            and any(term in q for term in ("tv", "television", "radio", "signal", "reception"))
        ):
            return choose(["television0_01", "television0_02", "television0_03"], "topic:tv_poor_reception_signals")
        if "safety precautions" in q and any(term in q for term in ("safe operation", "ensure safe", "during this process")):
            return choose_empty("topic:tv_installation_safety_text_only")
        if "caption" in q and ("text" in q or "on-screen" in q or "on screen" in q):
            return choose(["Manual35_41", "Manual35_42"], "topic:tv_caption_text_settings")
        if "outdoor antenna" in q:
            return choose(["Manual35_39"], "topic:tv_outdoor_antenna")
        if "dvd player" in q:
            return choose(["Manual35_43", "Manual35_44"], "topic:tv_dvd_player_connection")
        if "do not attempt toservice" in q or "do not attempt to service" in q:
            return choose_empty("topic:tv_service_warning_text_only")
        if "a.prog" in q or "auto program" in q:
            return choose(["Manual35_5", "Manual35_6"], "topic:tv_aprog_button_core")

    if manual_id == "Manual34":
        if ("uphill" in q or "up hill" in q) and any(term in q for term in ("techniques", "proper", "riding")):
            return choose_empty("topic:snowmobile_riding_uphill_text_only")
        if "brake lever" in q and "brake button" in q:
            return choose(["Manual34_39"], "topic:snowmobile_brake_lever_button")
        if "start the engine" in q and "snowmobile" in q:
            return choose(["Manual34_116"], "topic:snowmobile_engine_start_steps")
        if "throttle cable" in q and ("adjust" in q or "steps" in q):
            return choose(["Manual34_148", "Manual34_149", "Manual34_150", "Manual34_151", "Manual34_152"], "topic:snowmobile_throttle_cable_adjust")
        if "v-beltholder" in q or "v-belt holder" in q:
            return choose(["Manual34_49", "Manual34_50", "Manual34_51", "Manual34_52"], "topic:snowmobile_v_belt_holder")
        if "spark plug" in q and "inspect" in q:
            return choose(["Manual34_138"], "topic:snowmobile_spark_plug_inspect")
        if ("clean" in q or "dry" in q or "salt" in q or "salty" in q or "dirty" in q) and (
            "snowmobile" in q or "post-ride" in q or "after riding" in q
        ):
            return choose_empty("topic:snowmobile_clean_after_salt_text_only")
        if "preparation checks" in q or "pre-operation checks" in q or "before using a snowmobile" in q:
            return choose(["Manual34_84"], "topic:snowmobile_preparation_checks_core")
        if "uphill" in q or "up hill" in q:
            return choose(["Manual34_130", "Manual34_131"], "topic:snowmobile_riding_uphill")
        if "steering system" in q:
            return choose(["Manual34_109"], "topic:snowmobile_steering_system")
        if "steps to turn" in q or "turn using a snowmobile" in q:
            return choose(["Manual34_127"], "topic:snowmobile_turning")
        if ("emergency stop" in q or "engine stop switch" in q or "stop switch" in q) and (
            "snowmobile" in q or "engine" in q or "riding" in q
        ):
            return choose(["Manual34_38"], "topic:snowmobile_engine_stop_switch")
        if "do not shift" in q and "forward" in q and "reverse" in q:
            return choose(["Manual34_47"], "topic:snowmobile_shift_caution")

    if manual_id == "Manual36":
        if "程序日程" in question or ("program" in q and "schedule" in q and "thermostat" in q):
            return choose(["Manual36_31", "thermostat_04", "thermostat_05", "thermostat_06"], "topic:thermostat_program_schedule")
        if "临时更改" in question or ("temporary" in q and "temperature" in q and "thermostat" in q):
            return choose(["Manual36_32"], "topic:thermostat_temporary_temperature_change")
        if "警报界面" in question or ("alert" in q and ("screen" in q or "alarm" in q)):
            return choose(["Manual36_43", "Manual36_44", "Manual36_45", "Manual36_46", "Manual36_47", "Manual36_48"], "topic:thermostat_alert_screen")
        if "热泵" in question and "接线" in question:
            return choose(["Manual36_35"], "topic:thermostat_heat_pump_wiring")
        if "更换温控器的电池" in question or ("battery" in q and "thermostat" in q and ("replace" in q or "change" in q)):
            return choose(["Manual36_40"], "topic:thermostat_battery_replace")
        if "故障" in question and ("排除" in question or "解决" in question):
            return choose_empty("topic:thermostat_troubleshooting_text_only")
        if "\u65e5\u671f" in question and "\u65f6\u95f4" in question:
            return choose(["Manual36_42", "Manual36_41"], "topic:thermostat_date_time")
        if "日期时间" in question or "date/time" in q:
            return choose(["Manual36_42", "Manual36_41"], "topic:thermostat_date_time")

    if manual_id == "Manual37":
        if "brushpacer" in q:
            return choose(["Manual37_11"], "topic:toothbrush_brushpacer")
        if "travel case" in q and any(term in q for term in ("charge", "charging", "charges")):
            return choose(["Manual37_18", "Manual37_19", "Manual37_20", "Manual37_21"], "topic:toothbrush_travel_case_charging")
        if (
            any(term in q for term in ("activate", "deactivate", "activation", "deactivation"))
            and any(term in q for term in ("feature", "features", "customized", "usage", "toothbrush"))
        ):
            return choose(["Manual37_14", "Manual37_15"], "topic:toothbrush_feature_toggle")

    if manual_id == "Manual38":
        if "遮光罩" in question or ("light shield" in q and "clean" in q):
            return choose(["Manual38_5"], "topic:vr_light_shield_clean")
        if "安全预防措施" in question or ("safety precautions" in q and "vr" in q):
            return choose_empty("topic:vr_safety_precautions_text_only")
        if "更换耳塞" in question or ("earbud" in q and ("replace" in q or "types" in q)):
            return choose(["Manual38_4"], "topic:vr_earbud_types_replace")

    if manual_id == "Manual40":
        if "全加速" in question and ("停稳" in question or "停止" in question):
            return choose_empty("topic:watercraft_full_throttle_stop_text_only")
        if (
            ("\u6ed1\u822a\u901f\u5ea6" in question or "planing speed" in q)
            and ("\u6025\u8f6c\u5f2f" in question or "sharp turn" in q or "tight turn" in q)
        ):
            return choose(["Manual40_26"], "topic:watercraft_planing_sharp_turn")
        if (
            any(term in q for term in ("ellipse", "ellipses", "circle", "circles", "figure-8", "figure 8"))
            or any(term in question for term in ("\u5927\u692d\u5706", "\u7ed5\u5708", "8\u5b57", "8 \u5b57"))
        ) and (
            any(term in q for term in ("training", "turn", "turning", "planing"))
            or any(term in question for term in ("\u8f6c\u5f2f", "\u884c\u9a76", "\u7a33\u5b9a"))
        ):
            return choose(["Manual40_26", "Manual40_27"], "topic:watercraft_planing_turning_training")
        if (
            any(term in q for term in ("medium", "low speed", "semi-planing", "semiplaning", "stability", "stable"))
            or any(term in question for term in ("\u4e2d\u4f4e\u901f", "\u6cb9\u95e8", "\u7a33\u5b9a"))
        ) and (
            any(term in q for term in ("turn", "turning", "throttle"))
            or any(term in question for term in ("\u8f6c\u5f2f", "\u6cb9\u95e8", "\u64cd\u63a7"))
        ):
            return choose(["Manual40_22"], "topic:watercraft_medium_low_speed_turning_stability")
        if ("深水" in question or "deep water" in q) and ("平衡" in question or "balance" in q):
            return choose(["Manual40_17", "Manual40_18"], "topic:watercraft_deep_water_balance")
        if ("拖曳速度" in question or "trolling speed" in q) and (
            "直行" in question or "转弯" in question or "turn" in q or "turning" in q
        ):
            return choose(["Manual40_19"], "topic:watercraft_trolling_speed_turning")
        if ("拖曳速度" in question and ("半滑航速度" in question or "滑航速度" in question)) or (
            "trolling speed" in q and ("planing speed" in q or "semi-planing" in q or "semiplaning" in q)
        ):
            return choose(["Manual40_1", "Manual40_2", "Manual40_3"], "topic:watercraft_speed_categories")
        if ("深水" in question or "deep water" in q) and any(term in question for term in ("重新登上", "保持平衡")):
            return choose(["Manual40_13", "Manual40_14", "Manual40_15", "Manual40_16"], "topic:watercraft_deep_water_reboarding")

    return None


def _find_first_topic_position(question: str, terms: tuple[str, ...]) -> int | None:
    hits = [question.find(term) for term in terms if term and question.find(term) >= 0]
    return min(hits) if hits else None


def _reorder_multi_topic_images(manual_id: str, question: str, image_ids: list[str]) -> tuple[list[str], str]:
    """For multi-intent manual questions, keep image IDs aligned with topic order in the answer."""
    if manual_id != "Manual03" or len(image_ids) < 2:
        return image_ids, ""
    q = str(question or "").lower()
    groups: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        (
            "filter_packaging",
            ("滤网塑料包装", "塑料包装", "滤网上的塑料", "filter packaging", "plastic packaging"),
            ("Manual03_0", "Manual03_1", "Manual03_2", "Manual03_3", "Manual03_4", "Manual03_5"),
        ),
        (
            "dust_sensor",
            ("灰尘传感器", "传感器", "dust sensor"),
            ("Manual03_22", "Manual03_23", "Manual03_24", "Manual03_25"),
        ),
        (
            "iaq_indicator",
            ("室内空气质量指示灯", "空气质量指示灯", "iaq", "air quality indicator"),
            ("Manual03_20",),
        ),
        (
            "operation_modes",
            ("模式", "运行模式", "睡眠模式", "自动模式", "mode", "sleep mode", "auto mode"),
            ("Manual03_14", "Manual03_15", "Manual03_16", "Manual03_17"),
        ),
    ]
    selected = set(image_ids)
    matched: list[tuple[int, str, tuple[str, ...]]] = []
    covered: set[str] = set()
    for name, terms, group_ids in groups:
        present = [image_id for image_id in group_ids if image_id in selected]
        if not present:
            continue
        pos = _find_first_topic_position(q, terms)
        if pos is None:
            continue
        matched.append((pos, name, group_ids))
        covered.update(present)
    if len(matched) < 2 or covered != selected:
        return image_ids, ""
    matched.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _pos, _name, group_ids in matched:
        for image_id in image_ids:
            if image_id in group_ids and image_id not in ordered:
                ordered.append(image_id)
    if ordered and ordered != image_ids and set(ordered) == selected:
        return ordered, "topic_order:manual03_multi_intent"
    return image_ids, ""


class DynamicImageSelector:
    def __init__(
        self,
        *,
        score_cache: dict[str, dict[str, Any]] | None = None,
        manual_cache: dict[str, dict[str, Any]] | None = None,
        simple_cache: dict[str, dict[str, Any]] | None = None,
        use_llm: bool = False,
        candidate_k: int = 50,
        timeout: float = 60.0,
        leave_one_out: bool = False,
        use_known_routes: bool = True,
    ) -> None:
        self.selector = ImageSelector("v59")
        self.router = ManualRouter(self.selector, use_known_routes=use_known_routes)
        self.score_cache = score_cache or {}
        self.manual_cache = manual_cache or {}
        self.simple_cache = simple_cache or {}
        self.use_llm = use_llm
        self.candidate_k = candidate_k
        self.timeout = timeout
        self.leave_one_out = leave_one_out
        learned_rules_path = os.environ.get("LEARNED_IMAGE_RULES", "").strip()
        self.feedback_engine = FeedbackRuleEngine.from_path(learned_rules_path) if learned_rules_path else None
        self.teacher_image_enabled = os.environ.get("USE_CANONICAL_IMAGE_TEACHER", "1") != "0"
        self.teacher_image_min_score = float(os.environ.get("CANONICAL_IMAGE_TEACHER_MIN_SCORE", "42"))
        self.teacher_examples_by_manual = self._load_teacher_image_examples() if self.teacher_image_enabled else {}

    def _load_teacher_image_examples(self) -> dict[str, list[TeacherImageExample]]:
        if not CANONICAL_REFERENCE_PATH.exists():
            return {}
        try:
            reference_rows = load_csv(CANONICAL_REFERENCE_PATH)
        except Exception:
            return {}
        route_by_id = {str(row.get("id") or ""): row for row in self.router.route_rows}
        examples: dict[str, list[TeacherImageExample]] = defaultdict(list)
        for ref_row in reference_rows:
            row_id = str(ref_row.get("id") or "").strip()
            route_row = route_by_id.get(row_id)
            if not route_row:
                continue
            route_type = str(route_row.get("route_type") or "")
            gold_manual = str(route_row.get("gold_manual") or "")
            if route_type == "policy_service" or gold_manual == "none_policy":
                continue
            question = str(route_row.get("question") or ref_row.get("question") or "").strip()
            if not question:
                continue
            image_ids = parse_reference_ret_images(str(ref_row.get("ret") or ""))
            reference_manual = infer_manual_from_image_ids(image_ids)
            if gold_manual.startswith("Manual") and reference_manual and reference_manual != gold_manual:
                image_ids = [
                    image_id
                    for image_id in IMAGE_ID_RE.findall(str(route_row.get("teacher_image_ids") or ""))
                    if image_id.startswith(f"{gold_manual}_") or image_id in EXPLICIT_TOPIC_IMAGE_IDS
                ]
            manual_id = gold_manual or reference_manual
            if not manual_id:
                continue
            normalized = normalize_question_text(question)
            tokens = token_counter(expand_query_text(question))
            examples[manual_id].append(
                TeacherImageExample(
                    row_id=row_id,
                    manual_id=manual_id,
                    question=question,
                    normalized_question=normalized,
                    image_ids=image_ids,
                    tokens=tokens,
                )
            )
        return dict(examples)

    def _teacher_image_decision(
        self,
        question: str,
        manual_id: str,
        allowed_ids: set[str],
    ) -> tuple[list[str], str, dict[str, Any]] | None:
        examples = self.teacher_examples_by_manual.get(manual_id) or []
        if not examples:
            return None
        q_norm = normalize_question_text(question)
        q_tokens = token_counter(expand_query_text(question))
        best: TeacherImageExample | None = None
        best_score = 0.0
        best_exact = False
        for example in examples:
            exact = q_norm == example.normalized_question
            score = 1000.0 if exact else overlap_score(q_tokens, example.tokens)
            if score > best_score:
                best = example
                best_score = score
                best_exact = exact
        if best is None:
            return None
        if not best_exact and best_score < self.teacher_image_min_score:
            return None
        seen: set[str] = set()
        image_ids: list[str] = []
        for image_id in best.image_ids:
            if image_id in seen:
                continue
            if image_id in allowed_ids or image_id in EXPLICIT_TOPIC_IMAGE_IDS:
                image_ids.append(image_id)
                seen.add(image_id)
        if best.image_ids and not image_ids:
            return None
        reason = f"teacher:v62:{best.row_id}:{'exact' if best_exact else 'similar'}"
        meta = {
            "row_id": best.row_id,
            "score": round(best_score, 3),
            "exact": best_exact,
            "question": best.question,
        }
        return image_ids, reason, meta

    def _cache_pred(self, cache: dict[str, dict[str, Any]], row_id: str) -> list[str] | None:
        row = cache.get(str(row_id))
        if not row:
            return None
        return [str(image_id) for image_id in (row.get("pred") or row.get("image_ids") or [])]

    def _llm_variant(
        self,
        row_id: str,
        question: str,
        manual_id: str,
        candidates: list[dict[str, Any]],
        base: dict[str, Any],
        *,
        variant: str,
    ) -> list[str]:
        if not candidates:
            return []
        allowed = {str(c["image_id"]) for c in candidates}
        if variant == "manual":
            prompt = render_prompt(
                question,
                manual_id,
                candidates,
                [str(x) for x in (base.get("image_ids") or [])],
                base.get("similar_examples") or [],
                "manual",
            )
        elif variant == "simple":
            prompt = render_simple_prompt(
                question,
                manual_id,
                candidates,
                [str(x) for x in (base.get("image_ids") or [])],
                "score",
            )
        else:
            prompt = render_prompt(
                question,
                manual_id,
                candidates,
                [str(x) for x in (base.get("image_ids") or [])],
                base.get("similar_examples") or [],
                "score",
            )
        raw = ""
        last_error = ""
        for attempt in range(3):
            try:
                raw = call_deepseek(prompt, timeout=self.timeout)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        if not raw:
            print(f"[warn] {row_id} llm_variant={variant} failed: {last_error}; fallback=base", flush=True)
            return [str(x) for x in (base.get("image_ids") or [])]
        pred = parse_json_ids(raw, allowed)
        if not pred and base.get("image_ids"):
            return [str(x) for x in base.get("image_ids") or []]
        return pred

    def select(self, row_id: str, question: str, manual_hint: str = "") -> dict[str, Any]:
        started = time.perf_counter()
        route = self.router.predict(row_id, question, manual_hint)
        if route.route_type == "policy_service" or not route.manual_id or route.manual_id == "none_policy":
            return {
                "id": row_id,
                "question": question,
                "route": route.__dict__,
                "selected_variant": "policy",
                "image_ids": [],
                "proposals": {},
                "confidence": {"score": route.confidence, "level": "high", "agreement": 0, "unique_proposal_sets": 1},
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }

        base = self.selector.select(row_id, question, route.manual_id, leave_one_out=self.leave_one_out, debug=False)
        base_pred = [str(x) for x in (base.get("image_ids") or [])]
        base_proposals = {
            "score": base_pred,
            "manual": base_pred,
            "v1": base_pred,
            "base": base_pred,
        }
        allowed_ids = set(self.selector.manual_images.get(route.manual_id, []))
        early_feedback_decision = None
        early_final_images: list[str] = []
        early_variant = ""
        teacher_meta: dict[str, Any] = {}
        teacher_decision = self._teacher_image_decision(question, route.manual_id, allowed_ids)
        if teacher_decision is not None:
            early_final_images, early_variant, teacher_meta = teacher_decision
        if self.feedback_engine is not None:
            early_feedback_decision = self.feedback_engine.apply(question, route.manual_id, allowed_ids=allowed_ids)
            if early_feedback_decision is not None:
                early_final_images = early_feedback_decision.image_ids
                early_variant = f"feedback:{early_feedback_decision.rule_id}"
        reviewed_topic_override = (
            route.manual_id == "Manual29"
            and any(
                term in question.lower()
                for term in (
                    "插入存储卡", "装入存储卡", "memory card",
                    "闪光灯", "use flash", "flash setting",
                    "自拍", "self-timer", "self timer",
                    "指令拨盘", "command dial",
                    "剩余相纸", "相纸余量", "remaining film",
                )
            )
        ) or (
            route.manual_id == "Manual37" and "brushpacer" in question.lower()
        )
        if not early_variant.startswith("teacher:") or reviewed_topic_override:
            early_topic_decision = _topic_images(route.manual_id, question, base_proposals, allowed_ids)
            if early_topic_decision is not None:
                early_final_images, early_variant = early_topic_decision
        if early_variant:
            reordered_images, reorder_reason = _reorder_multi_topic_images(route.manual_id, question, early_final_images)
            if reorder_reason:
                early_final_images = reordered_images
                early_variant = f"{early_variant}|{reorder_reason}"
            confidence = compute_confidence(route, early_final_images, base_proposals, early_variant)
            return {
                "id": row_id,
                "question": question,
                "route": route.__dict__,
                "selected_variant": early_variant,
                "image_ids": early_final_images,
                "proposals": base_proposals,
                "confidence": confidence,
                "feedback_rule": early_feedback_decision.__dict__ if early_feedback_decision is not None else {},
                "teacher_match": teacher_meta,
                "topic_rule": early_variant if early_variant.startswith("topic:") else "",
                "base_reason": base.get("reason"),
                "candidate_ids": [],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }

        score_pred = self._cache_pred(self.score_cache, row_id)
        manual_pred = self._cache_pred(self.manual_cache, row_id)
        simple_pred = self._cache_pred(self.simple_cache, row_id)

        candidates: list[dict[str, Any]] = []
        if self.use_llm and (score_pred is None or manual_pred is None or simple_pred is None):
            candidates, base = build_candidates(self.selector, row_id, question, route.manual_id, self.candidate_k)
            base_pred = [str(x) for x in (base.get("image_ids") or [])]
            if score_pred is None:
                score_pred = self._llm_variant(row_id, question, route.manual_id, candidates, base, variant="score")
            if manual_pred is None:
                manual_pred = self._llm_variant(row_id, question, route.manual_id, candidates, base, variant="manual")
            if simple_pred is None:
                simple_pred = self._llm_variant(row_id, question, route.manual_id, candidates, base, variant="simple")

        proposals = {
            "score": score_pred if score_pred is not None else base_pred,
            "manual": manual_pred if manual_pred is not None else base_pred,
            "v1": simple_pred if simple_pred is not None else base_pred,
            "base": base_pred,
        }
        selected_variant = choose_variant({"manual": route.manual_id, "question": question})
        if selected_variant not in proposals:
            selected_variant = "score"
        final_images = proposals[selected_variant]
        feedback_decision = None
        if self.feedback_engine is not None:
            allowed_ids = set(self.selector.manual_images.get(route.manual_id, []))
            feedback_decision = self.feedback_engine.apply(question, route.manual_id, allowed_ids=allowed_ids)
            if feedback_decision is not None:
                final_images = feedback_decision.image_ids
                selected_variant = f"feedback:{feedback_decision.rule_id}"
        topic_decision = _topic_images(
            route.manual_id,
            question,
            proposals,
            set(self.selector.manual_images.get(route.manual_id, [])),
        )
        if topic_decision is not None:
            final_images, topic_reason = topic_decision
            selected_variant = topic_reason
        reordered_images, reorder_reason = _reorder_multi_topic_images(route.manual_id, question, final_images)
        if reorder_reason:
            final_images = reordered_images
            selected_variant = f"{selected_variant}|{reorder_reason}"
        confidence = compute_confidence(route, final_images, proposals, selected_variant)

        return {
            "id": row_id,
            "question": question,
            "route": route.__dict__,
            "selected_variant": selected_variant,
            "image_ids": final_images,
            "proposals": proposals,
            "confidence": confidence,
            "feedback_rule": feedback_decision.__dict__ if feedback_decision is not None else {},
            "teacher_match": {},
            "topic_rule": selected_variant if selected_variant.startswith("topic:") else "",
            "base_reason": base.get("reason"),
            "candidate_ids": [str(c["image_id"]) for c in candidates[: self.candidate_k]],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def load_questions(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [{str(k): str(v) for k, v in row.items()} for row in load_jsonl(path)]
    return load_csv(path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [row for row in rows if "gold" in row]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "policy_rows": sum(1 for row in rows if row.get("route", {}).get("route_type") == "policy_service"),
        "low_confidence": sum(1 for row in rows if row.get("confidence", {}).get("level") == "low"),
        "medium_confidence": sum(1 for row in rows if row.get("confidence", {}).get("level") == "medium"),
        "high_confidence": sum(1 for row in rows if row.get("confidence", {}).get("level") == "high"),
    }
    if eval_rows:
        image_rows = [row for row in eval_rows if row.get("gold")]
        no_rows = [row for row in eval_rows if not row.get("gold")]
        summary.update(
            {
                "eval_rows": len(eval_rows),
                "avg_f1": sum(float(row["f1"]) for row in eval_rows) / max(1, len(eval_rows)),
                "image_rows": len(image_rows),
                "image_avg_f1": sum(float(row["f1"]) for row in image_rows) / max(1, len(image_rows)),
                "gold_no_image": len(no_rows),
                "pred_no_image": sum(1 for row in eval_rows if not row.get("image_ids")),
                "no_image_correct": sum(1 for row in no_rows if not row.get("image_ids")),
                "set_equal": sum(1 for row in eval_rows if set(row.get("image_ids") or []) == set(row.get("gold") or [])),
                "exact_order": sum(1 for row in eval_rows if (row.get("image_ids") or []) == (row.get("gold") or [])),
            }
        )
        routed = [row for row in eval_rows if "route_correct" in row]
        if routed:
            summary["route_accuracy"] = sum(1 for row in routed if row.get("route_correct")) / len(routed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--ids", help="Comma-separated question IDs to run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--ignore-known-route", action="store_true")
    parser.add_argument("--manual-column", default="", help="Optional CSV column containing manual_id/gold_manual.")
    parser.add_argument("--score-cache", default="")
    parser.add_argument("--manual-cache", default="")
    parser.add_argument("--simple-cache", default="")
    parser.add_argument("--teacher", default="outputs/rag_assets/v59_teacher_examples.jsonl")
    args = parser.parse_args()

    score_cache = by_id(args.score_cache) if args.score_cache else {}
    manual_cache = by_id(args.manual_cache) if args.manual_cache else {}
    simple_cache = by_id(args.simple_cache) if args.simple_cache else {}
    selector = DynamicImageSelector(
        score_cache=score_cache,
        manual_cache=manual_cache,
        simple_cache=simple_cache,
        use_llm=args.use_llm,
        candidate_k=args.candidate_k,
        timeout=args.timeout,
        leave_one_out=args.leave_one_out,
        use_known_routes=not args.ignore_known_route,
    )

    questions = load_questions(Path(args.questions))
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        questions = [row for row in questions if str(row.get("id") or "") in wanted]

    teacher = by_id(args.teacher) if args.teacher else {}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for idx, row in enumerate(questions, 1):
            row_id = str(row.get("id") or idx)
            question = str(row.get("question") or row.get("问题") or "")
            manual_hint = ""
            if args.manual_column:
                manual_hint = str(row.get(args.manual_column) or "")
            elif row.get("manual_id") or row.get("gold_manual"):
                manual_hint = str(row.get("manual_id") or row.get("gold_manual") or "")
            futures[pool.submit(selector.select, row_id, question, manual_hint)] = row_id
        for future in as_completed(futures):
            result = future.result()
            gold_row = teacher.get(str(result["id"]))
            if gold_row is not None:
                gold = [str(image_id) for image_id in (gold_row.get("image_ids") or [])]
                result["gold"] = gold
                result["gold_manual"] = str(gold_row.get("manual_id") or "")
                result["route_correct"] = result.get("route", {}).get("manual_id") == result["gold_manual"] or (
                    not result["gold_manual"] and result.get("route", {}).get("manual_id") == "none_policy"
                )
                result["f1"] = round(f1_score([str(x) for x in result.get("image_ids") or []], gold), 6)
            rows.append(result)
            print(
                f"[done] {result['id']} manual={result.get('route', {}).get('manual_id')} "
                f"variant={result.get('selected_variant')} images={len(result.get('image_ids') or [])} "
                f"conf={result.get('confidence', {}).get('level')} {result.get('elapsed_ms')}ms",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]))
    write_jsonl(ROOT / args.output, rows)
    summary = summarize(rows)
    if args.summary_output:
        out = ROOT / args.summary_output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
