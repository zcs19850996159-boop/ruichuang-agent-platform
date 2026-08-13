from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "rag_agent"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_ret(answer: str, images: list[str]) -> str:
    image_list = "[" + ", ".join(json.dumps(str(image_id), ensure_ascii=False) for image_id in images) + "]"
    return f"\"{(answer or '').strip().strip(chr(34))}\", {image_list}"


def main() -> None:
    rows: dict[str, dict] = {}
    base = RESULT_DIR / "question_public_strict_v59_direct_deepseek_results.jsonl"
    retry_paths = [
        RESULT_DIR / "strict_v59_retry5_results.jsonl",
        RESULT_DIR / "strict_v59_retry417_results.jsonl",
    ]
    for row in load_jsonl(base):
        rows[str(row["id"])] = row
    for path in retry_paths:
        for row in load_jsonl(path):
            rows[str(row["id"])] = row

    ordered = [rows[key] for key in sorted(rows, key=lambda value: int(value))]
    out_results = RESULT_DIR / "question_public_strict_v59_direct_deepseek_rechecked_results.jsonl"
    out_csv = RESULT_DIR / "question_public_strict_v59_direct_deepseek_rechecked_submission.csv"

    with out_results.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "ret"])
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "ret": format_ret(row.get("answer") or "", row.get("images") or []),
                }
            )
    print(f"merged={len(ordered)}")
    print(out_results)
    print(out_csv)


if __name__ == "__main__":
    main()
