from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item).strip()))


def main() -> None:
    raw_path = ASSET_DIR / "question_review_image_overrides.jsonl"
    rows = load_jsonl(raw_path)
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row["id"])
        action = row.get("review_action") or ""
        question = row.get("question") or ""
        current = dedupe(row.get("current_images") or [])
        selected = dedupe(row.get("selected_images") or [])
        notes = []

        if not selected and current and action != "修跨手册图":
            selected = current
            notes.append("fallback_current_when_model_pruned_to_zero")

        # Hard corrections from the review workbook.
        hard_overrides = {
            "64": ["Manual04_3"],
            "101": ["Manual06_14", "Manual06_15"],
            "249": ["Manual09_161", "Manual09_162", "Manual09_163", "Manual09_164", "Manual09_165", "Manual09_166"],
            "289": ["Camera_31", "Manual10_100", "Manual10_101", "Manual10_102", "Manual10_103"],
            "313": ["Manual15_7", "Manual15_8", "Manual15_9", "fax_08", "Manual15_10", "Manual15_11", "Manual15_12", "Manual15_15"],
            "411": ["Manual32_5", "Manual32_6"],
            "225": ["oven_13"],
            "226": ["oven_14"],
            "227": ["oven_15"],
        }
        # Earlier manual QA diagnostics showed these English/figure-heavy rows lose
        # substantial content if they are collapsed to a single generic figure.
        preserve_current = {"115", "249", "259", "289", "313", "372", "411"}
        if rid in preserve_current and current:
            selected = current
            notes.append("preserved_current_for_figure_heavy_question")

        if rid in hard_overrides:
            selected = hard_overrides[rid]
            notes.append("hard_review_override")

        if rid == "80":
            selected = [image_id for image_id in selected if image_id not in {"air_conditioner_01", "Manual01_5", "Manual01_6"}]
            if not selected:
                selected = ["Manual01_22", "Manual01_23"]
            notes.append("removed_remote_images_for_no_remote_operation")

        if rid == "161":
            selected = [image_id for image_id in selected if image_id != "Manual18_71"]
            notes.append("removed_Manual18_71_from_identification_question")

        if rid == "172":
            selected = []
            notes.append("removed_all_images_for_precision_device_text_note")

        if "Manual18_71" in selected and not any(term in question for term in ("技术参数", "规格", "AC", "DC", "输出")):
            selected = [image_id for image_id in selected if image_id != "Manual18_71"]
            notes.append("removed_Manual18_71_outside_output_specs")

        if action == "修跨手册图":
            manual_id = row.get("manual_id") or ""
            selected = [image_id for image_id in selected if image_id.startswith(manual_id + "_")]
            notes.append("removed_cross_manual_images")

        row["raw_selected_images"] = row.get("selected_images") or []
        row["selected_images"] = dedupe(selected)
        row["postprocess_notes"] = notes
        final_rows.append(row)

    out = ASSET_DIR / "question_review_image_overrides_final.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "rows": len(final_rows),
        "zero_with_current": sum(1 for row in final_rows if row.get("current_images") and not row.get("selected_images")),
        "changed": sum(1 for row in final_rows if row.get("selected_images") != row.get("current_images")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
