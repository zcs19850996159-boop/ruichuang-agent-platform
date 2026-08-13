from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
ASSET_DIR = ROOT / "outputs" / "rag_assets"
ROUTE_PATH = ROOT / "work" / "a_rank_question_route_gold.csv"
QUESTION_PATH = DOWNLOADS / "question_public (2).csv"


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def load_routes() -> dict[str, dict[str, str]]:
    with ROUTE_PATH.open(encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def parse_ret(ret: str) -> tuple[str, list[str], bool]:
    text = (ret or "").strip()
    match = re.search(r",\s*(\[[^\]]*\])\s*$", text, re.S)
    if not match:
        return text.strip().strip('"').strip(), [], False
    answer = text[: match.start()].strip().strip('"').strip()
    raw_images = match.group(1)
    try:
        parsed = json.loads(raw_images)
    except Exception:
        parsed = ast.literal_eval(raw_images)
    images = [str(item).strip() for item in parsed if str(item).strip()] if isinstance(parsed, list) else []
    return answer, images, True


def is_english(text: str) -> bool:
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return letters >= 3 and letters > cjk


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Teacher submission CSV with id,ret columns.")
    parser.add_argument("--tag", required=True, help="Asset tag, for example v59.")
    parser.add_argument("--questions", default=str(QUESTION_PATH))
    args = parser.parse_args()

    source_path = Path(args.source)
    refs = load_csv_by_id(source_path)
    questions = load_csv_by_id(Path(args.questions))
    routes = load_routes()

    rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    for rid in sorted(refs, key=lambda value: int(value) if value.isdigit() else value):
        answer, images, has_image_list = parse_ret(refs[rid].get("ret", ""))
        question = (questions.get(rid, {}).get("question") or refs[rid].get("question") or "").strip()
        route = routes.get(rid, {})
        pic_count = answer.count("<PIC>")
        row = {
            "id": rid,
            "question": question,
            "answer": answer,
            "answer_sample_plain": answer,
            "image_ids": images,
            "ret": refs[rid].get("ret", ""),
            "has_image_list": has_image_list,
            "pic_count": pic_count,
            "image_count": len(images),
            "pic_image_aligned": pic_count == len(images),
            "language": "en" if is_english(question) else "zh",
            "route_type": route.get("route_type", ""),
            "manual_id": route.get("gold_manual", ""),
            "intent_type": route.get("intent_type", ""),
            "image_need": route.get("image_need", ""),
            "source": str(source_path),
            "teacher_tag": args.tag,
        }
        rows.append(row)
        sft_rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a multimodal customer-service manual agent. "
                            "Answer in the user's language and place <PIC> exactly where selected images belong."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Manual: {row['manual_id']}\n"
                            f"Selected images: {images}"
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "id": rid,
                    "manual_id": row["manual_id"],
                    "language": row["language"],
                    "image_ids": images,
                    "has_image_list": has_image_list,
                    "pic_count": pic_count,
                    "teacher_tag": args.tag,
                },
            }
        )

    write_jsonl(ASSET_DIR / f"{args.tag}_teacher_examples.jsonl", rows)
    write_jsonl(ASSET_DIR / f"{args.tag}_sft_messages.jsonl", sft_rows)
    stats = {
        "source": str(source_path),
        "tag": args.tag,
        "examples": len(rows),
        "has_image_list": sum(1 for row in rows if row["has_image_list"]),
        "without_image_list": sum(1 for row in rows if not row["has_image_list"]),
        "pic_image_aligned": sum(1 for row in rows if row["pic_image_aligned"]),
        "pic_image_mismatched": sum(1 for row in rows if not row["pic_image_aligned"]),
        "by_language": Counter(row["language"] for row in rows),
        "by_manual": Counter(row["manual_id"] for row in rows),
        "image_count_hist": Counter(row["image_count"] for row in rows),
        "pic_count_hist": Counter(row["pic_count"] for row in rows),
    }
    (ASSET_DIR / f"{args.tag}_teacher_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
