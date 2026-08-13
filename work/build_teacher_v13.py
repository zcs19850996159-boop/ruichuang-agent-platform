from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_PATH = Path(r"C:/Users/admin/Downloads/reference_teacher_v12_p1_manual14_16_plus_on_v11 (1).csv")
QUESTION_PATH = Path(r"C:/Users/admin/Downloads/question_public (2).csv")
OUT_DIR = ROOT / "outputs" / "rag_agent"
PRESERVE_OUT = OUT_DIR / "question_public_teacher_v12_preserve_submission.csv"
FIXED_OUT = OUT_DIR / "question_public_teacher_v13_formatfixed_submission.csv"
HYBRID_OUT = OUT_DIR / "question_public_teacher_v13_hybrid_submission.csv"
REPORT_OUT = OUT_DIR / "question_public_teacher_v13_format_report.jsonl"


IMAGE_OVERRIDES = {
    "128": ["Manual11_7", "drill0_14"],
    "225": ["oven_13"],
    "226": ["oven_14"],
    "227": ["oven_15"],
}


def strip_outer_quotes(text: str) -> str:
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def parse_ret(ret: str) -> tuple[str, list[str], bool]:
    ret = ret or ""
    match = re.search(r",\s*(\[[^\]]*\])\s*$", ret, re.S)
    if not match:
        return strip_outer_quotes(ret), [], False
    answer = strip_outer_quotes(ret[: match.start()])
    try:
        images = ast.literal_eval(match.group(1))
    except Exception:
        images = []
    return answer, [str(image_id) for image_id in images], True


def format_ret(answer: str, images: list[str]) -> str:
    image_list = "[" + ", ".join(json.dumps(image_id, ensure_ascii=False) for image_id in images) + "]"
    return f'"{answer}", {image_list}'


def normalize_pic_count(answer: str, images: list[str]) -> tuple[str, str]:
    pic_count = answer.count("<PIC>")
    image_count = len(images)
    if pic_count == image_count:
        return answer, "unchanged"
    if image_count > pic_count:
        missing = image_count - pic_count
        suffix = "<PIC>" * missing
        if answer.rstrip().endswith("<PIC>"):
            return answer.rstrip() + suffix, f"added_{missing}_pic"
        return answer.rstrip() + "\n" + suffix, f"added_{missing}_pic"
    extra = pic_count - image_count
    fixed = answer
    for _ in range(extra):
        idx = fixed.rfind("<PIC>")
        if idx < 0:
            break
        fixed = fixed[:idx] + fixed[idx + len("<PIC>") :]
    fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip()
    return fixed, f"removed_{extra}_pic"


def load_known_image_ids() -> set[str]:
    known: set[str] = set()
    for path in [ROOT / "outputs" / "rag_assets" / "pic_mapping.jsonl", ROOT / "outputs" / "rag_assets" / "english_pic_captions.jsonl"]:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            obj = json.loads(line)
            image_id = obj.get("image_id") or obj.get("pic_id") or obj.get("id")
            if image_id:
                known.add(str(image_id))
    return known


def main() -> None:
    questions = {
        row["id"]: row["question"]
        for row in csv.DictReader(QUESTION_PATH.open(encoding="utf-8-sig", newline=""))
    }
    known_images = load_known_image_ids()
    ref_rows = list(csv.DictReader(REF_PATH.open(encoding="utf-8-sig", newline="")))

    report_rows: list[dict[str, object]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (
        PRESERVE_OUT.open("w", encoding="utf-8-sig", newline="") as preserve_f,
        FIXED_OUT.open("w", encoding="utf-8-sig", newline="") as fixed_f,
        HYBRID_OUT.open("w", encoding="utf-8-sig", newline="") as hybrid_f,
    ):
        preserve_writer = csv.DictWriter(preserve_f, fieldnames=["id", "question", "ret"])
        fixed_writer = csv.DictWriter(fixed_f, fieldnames=["id", "question", "ret"])
        hybrid_writer = csv.DictWriter(hybrid_f, fieldnames=["id", "question", "ret"])
        preserve_writer.writeheader()
        fixed_writer.writeheader()
        hybrid_writer.writeheader()
        for row in ref_rows:
            row_id = row["id"]
            question = questions[row_id]
            answer, images, had_suffix = parse_ret(row["ret"])
            preserve_ret = format_ret(answer, images)
            fixed_answer, action = normalize_pic_count(answer, images)
            fixed_ret = format_ret(fixed_answer, images)
            hybrid_images = IMAGE_OVERRIDES.get(row_id, images)
            hybrid_answer, hybrid_action = normalize_pic_count(answer, hybrid_images)
            hybrid_ret = format_ret(hybrid_answer, hybrid_images)
            preserve_writer.writerow({"id": row_id, "question": question, "ret": preserve_ret})
            fixed_writer.writerow({"id": row_id, "question": question, "ret": fixed_ret})
            hybrid_writer.writerow({"id": row_id, "question": question, "ret": hybrid_ret})
            report_rows.append(
                {
                    "id": row_id,
                    "had_suffix": had_suffix,
                    "original_pic_count": answer.count("<PIC>"),
                    "image_count": len(images),
                    "action": action,
                    "hybrid_image_count": len(hybrid_images),
                    "hybrid_action": hybrid_action,
                    "unknown_images": [image_id for image_id in images if image_id not in known_images],
                    "hybrid_unknown_images": [image_id for image_id in hybrid_images if image_id not in known_images],
                }
            )

    with REPORT_OUT.open("w", encoding="utf-8") as report_f:
        for row in report_rows:
            report_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"preserve={PRESERVE_OUT}")
    print(f"fixed={FIXED_OUT}")
    print(f"hybrid={HYBRID_OUT}")
    print(f"report={REPORT_OUT}")


if __name__ == "__main__":
    main()
