from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STOP_TERMS = {
    "what",
    "when",
    "where",
    "which",
    "should",
    "could",
    "would",
    "follow",
    "manual",
    "according",
    "using",
    "before",
    "after",
    "correct",
    "proper",
    "procedure",
    "steps",
    "如何",
    "怎么",
    "应该",
    "哪些",
    "什么",
    "手册",
    "使用",
    "操作",
    "需要",
    "正确",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def f1_score(pred: list[str], gold: list[str]) -> float:
    ps = set(pred)
    gs = set(gold)
    tp = len(ps & gs)
    precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def terms_from_question(question: str, *, max_terms: int = 5) -> list[str]:
    q = str(question or "")
    low = q.lower()
    terms: list[str] = []
    for phrase in re.findall(r"[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2}", low):
        phrase = phrase.strip()
        if len(phrase) < 4:
            continue
        if phrase in STOP_TERMS:
            continue
        if any(part in STOP_TERMS for part in phrase.split()) and len(phrase.split()) == 1:
            continue
        terms.append(phrase)
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}", q):
        if token in STOP_TERMS:
            continue
        terms.append(token)
    counts = Counter(terms)
    ranked = sorted(counts, key=lambda term: (counts[term], len(term)), reverse=True)
    out: list[str] = []
    for term in ranked:
        if any(term != prev and term in prev for prev in out):
            continue
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


@dataclass
class FeedbackDecision:
    rule_id: str
    action: str
    image_ids: list[str]
    reason: str


class FeedbackRuleEngine:
    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self.rules = [rule for rule in rules if rule.get("status", "active") != "disabled"]

    @classmethod
    def from_path(cls, path: str | Path) -> "FeedbackRuleEngine":
        return cls(load_jsonl(path))

    def match_score(self, rule: dict[str, Any], question: str, manual_id: str) -> int:
        if rule.get("manual_id") not in {"*", manual_id}:
            return 0
        q = str(question or "").lower()
        terms = [str(term).lower() for term in rule.get("trigger_terms") or [] if str(term).strip()]
        if not terms:
            return 0
        hits = sum(1 for term in terms if term in q)
        if hits < int(rule.get("min_matches") or min(2, len(terms))):
            return 0
        return hits

    def apply(self, question: str, manual_id: str, allowed_ids: set[str] | None = None) -> FeedbackDecision | None:
        matches: list[tuple[int, int, dict[str, Any]]] = []
        for idx, rule in enumerate(self.rules):
            score = self.match_score(rule, question, manual_id)
            if score:
                matches.append((score, int(rule.get("support") or 1), rule))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        rule = matches[0][2]
        action = str(rule.get("action") or "")
        if action == "force_no_image":
            return FeedbackDecision(str(rule.get("rule_id") or ""), action, [], str(rule.get("rationale") or "feedback rule"))
        if action == "prefer_images":
            image_ids = [str(x) for x in (rule.get("image_ids") or []) if str(x).strip()]
            if allowed_ids is not None:
                image_ids = [image_id for image_id in image_ids if image_id in allowed_ids]
            if image_ids:
                return FeedbackDecision(str(rule.get("rule_id") or ""), action, image_ids, str(rule.get("rationale") or "feedback rule"))
        return None


def learn_rules(selector_rows: list[dict[str, Any]], teacher_rows: list[dict[str, Any]], *, max_rules: int = 200) -> list[dict[str, Any]]:
    teacher_by_id = {str(row.get("id")): row for row in teacher_rows}
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selector_rows:
        row_id = str(row.get("id") or "")
        teacher = teacher_by_id.get(row_id)
        if not teacher:
            continue
        manual_id = str(teacher.get("manual_id") or row.get("route", {}).get("manual_id") or "")
        if not manual_id or manual_id == "none_policy":
            continue
        pred = [str(x) for x in (row.get("image_ids") or row.get("pred") or [])]
        gold = [str(x) for x in (teacher.get("image_ids") or row.get("gold") or [])]
        score = f1_score(pred, gold)
        if score >= 0.999:
            continue
        question = str(row.get("question") or teacher.get("question") or "")
        terms = terms_from_question(question)
        if not terms:
            continue
        action = "force_no_image" if not gold else "prefer_images"
        key = (manual_id, action, tuple(gold), tuple(terms[:4]))
        item = grouped.setdefault(
            key,
            {
                "manual_id": manual_id,
                "action": action,
                "image_ids": gold,
                "trigger_terms": terms[:4],
                "min_matches": min(2, len(terms[:4])),
                "support": 0,
                "evidence_ids": [],
                "before_f1_values": [],
                "rationale": "",
                "status": "candidate",
            },
        )
        item["support"] += 1
        item["evidence_ids"].append(row_id)
        item["before_f1_values"].append(round(score, 6))
    rules = list(grouped.values())
    rules.sort(key=lambda rule: (rule["support"], -sum(rule["before_f1_values"]) / max(1, len(rule["before_f1_values"]))), reverse=True)
    for idx, rule in enumerate(rules[:max_rules], 1):
        avg_before = sum(rule.pop("before_f1_values")) / max(1, rule["support"])
        rule["rule_id"] = f"learned_image_rule_{idx:04d}"
        rule["avg_before_f1"] = round(avg_before, 6)
        if not rule["rationale"]:
            if rule["action"] == "force_no_image":
                rule["rationale"] = "Human/eval feedback indicates this intent should not attach manual figures."
            else:
                rule["rationale"] = "Human/eval feedback prefers this figure set for the matched manual intent."
    return rules[:max_rules]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-eval", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rules", type=int, default=200)
    args = parser.parse_args()

    rules = learn_rules(load_jsonl(args.selector_eval), load_jsonl(args.teacher), max_rules=args.max_rules)
    write_jsonl(args.output, rules)
    print(json.dumps({"rules": len(rules), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
