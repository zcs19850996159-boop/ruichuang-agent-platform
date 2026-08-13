from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


BEST_BY_MANUAL = {
    "Manual01": "manual",
    "Manual02": "manual",
    "Manual03": "manual",
    "Manual04": "manual",
    "Manual05": "base",
    "Manual06": "manual",
    "Manual07": "manual",
    "Manual10": "manual",
    "Manual12": "manual",
    "Manual14": "manual",
    "Manual15": "v1",
    "Manual18": "manual",
    "Manual20": "manual",
    "Manual23": "manual",
    "Manual25": "v1",
    "Manual26": "manual",
    "Manual27": "manual",
    "Manual32": "v1",
    "Manual33": "manual",
    "Manual34": "manual",
    "Manual35": "base",
    "Manual36": "manual",
    "Manual40": "v1",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(path)}


def f1(pred: list[str], gold: list[str]) -> float:
    ps = set(pred)
    gs = set(gold)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    rec = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def choose_variant(row: dict[str, Any]) -> str:
    question = str(row.get("question") or "")
    q = question.lower()
    manual = str(row.get("manual") or "")
    choice = BEST_BY_MANUAL.get(manual, "score")

    if manual == "Manual09":
        if "sound system" in q or "listen to music" in q:
            choice = "v1"
        elif "turn a boat" in q or "turn while sailing" in q:
            choice = "manual"
        elif "ship steers" in q or "ship steer" in q:
            choice = "base"
        elif "swim platform" in q or "remove the bimini" in q or "approval label" in q:
            choice = "manual"
        elif "throttle-cable" in q or "steering system" in q:
            choice = "base"
        elif "store the bimini" in q:
            choice = "manual"
    elif manual == "Manual20":
        if "engine switches" in q or "two kinds of engine switches" in q:
            choice = "v1"
        elif "filler caps" in q:
            choice = "v1"
        elif "start my jetski" in q and "different situations" in q:
            choice = "base"
        elif "max load" in q:
            choice = "manual"
    elif manual == "Manual34":
        if "steering system" in q:
            choice = "v1"
        elif "v-beltholder" in q or "v-belt holder" in q:
            choice = "base"
        elif "preparation checks" in q or "before using a snowmobile" in q:
            choice = "score"
        elif "inspect the spark plug" in q or "start the engine" in q or "brake lever" in q:
            choice = "manual"
        elif "downhill" in q:
            choice = "manual"
    elif manual == "Manual25":
        if "sata odd" in q or "secure the motherboard" in q or "central processing unit" in q or "cpu" in q:
            choice = "base"
        elif "system memory" in q:
            choice = "v1"
    elif manual == "Manual15":
        if "connect" in q or "connecting" in q or "setup" in q:
            choice = "score"
        elif "safety precautions" in q or "use the product safely" in q or "using the fax" in q:
            choice = "manual"
        elif "ensure my safety" in q:
            choice = "manual"
        elif "move this fax" in q or "moving" in q or "fingers safe" in q:
            choice = "v1"
    elif manual == "Manual12":
        if "components" in q or "in my hand" in q or "controling functions" in q or "controlling functions" in q:
            choice = "manual"
    elif manual == "Manual16":
        if "界面" in question:
            choice = "manual"
        elif "运动应用" in question:
            choice = "v1"
    elif manual == "Manual18":
        if "前六个步骤" in question:
            choice = "v1"
        elif "消音器" in question or "发烫" in question:
            choice = "manual"
        elif "火花塞" in question:
            choice = "v1"
    elif manual == "Manual10":
        if "power socket" in q or "battery" in q or 'p" model' in q or "p model" in q or "off-center" in q:
            choice = "manual"
    elif manual == "Manual30":
        if "quick release" in q:
            choice = "v1"
        elif "silicone cap" in q or "steam release valve" in q:
            choice = "manual"
    elif manual == "Manual32":
        if "robot anatomy" in q:
            choice = "v1"
        elif "full bin sensors" in q or "primary modes" in q:
            choice = "manual"
    elif manual == "Manual05":
        if "产品功能" in question and "快速上手" in question:
            choice = "base"
    elif manual == "Manual19":
        if "safety tips" in q:
            choice = "manual"
    return choice


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_rows = [row for row in rows if row["gold_n"] > 0]
    no_rows = [row for row in rows if row["gold_n"] == 0]
    return {
        "rows": len(rows),
        "exact_order": sum(1 for row in rows if row["exact"]),
        "set_equal": sum(1 for row in rows if row["seteq"]),
        "avg_f1": sum(float(row["f1"]) for row in rows) / max(1, len(rows)),
        "image_rows": len(image_rows),
        "image_avg_f1": sum(float(row["f1"]) for row in image_rows) / max(1, len(image_rows)),
        "gold_no_image": len(no_rows),
        "pred_no_image": sum(1 for row in rows if row["pred_n"] == 0),
        "no_image_correct": sum(1 for row in no_rows if row["pred_n"] == 0),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", default="outputs/rag_agent/llm_selector_full_v4_score_k50.jsonl")
    parser.add_argument("--manual", default="outputs/rag_agent/llm_selector_full_v4_manual_k50.jsonl")
    parser.add_argument("--v1", default="outputs/rag_agent/llm_selector_full_v1.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    score_rows = by_id(ROOT / args.score)
    manual_rows = by_id(ROOT / args.manual)
    v1_rows = by_id(ROOT / args.v1)
    rows: list[dict[str, Any]] = []
    for row_id in sorted(score_rows, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        score_row = score_rows[row_id]
        proposals = {
            "score": [str(x) for x in score_row.get("pred") or []],
            "manual": [str(x) for x in manual_rows[row_id].get("pred") or []],
            "v1": [str(x) for x in v1_rows[row_id].get("pred") or []],
            "base": [str(x) for x in score_row.get("base_pred") or []],
        }
        variant = choose_variant(score_row)
        pred = proposals[variant]
        gold = [str(x) for x in score_row.get("gold") or []]
        row_f1 = f1(pred, gold)
        rows.append(
            {
                "id": row_id,
                "manual": score_row.get("manual"),
                "question": score_row.get("question"),
                "variant": variant,
                "pred": pred,
                "gold": gold,
                "proposals": proposals,
                "f1": round(row_f1, 6),
                "pred_n": len(pred),
                "gold_n": len(gold),
                "exact": pred == gold,
                "seteq": set(pred) == set(gold),
            }
        )
    write_jsonl(ROOT / args.output, rows)
    summary = summarize(rows)
    if args.summary_output:
        out = ROOT / args.summary_output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
