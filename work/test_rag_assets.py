from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "rag_assets"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest_path = OUTPUT_DIR / "manifest.json"
    pic_path = OUTPUT_DIR / "pic_mapping.jsonl"
    chunks_path = OUTPUT_DIR / "manual_chunks.jsonl"
    risks_path = OUTPUT_DIR / "risk_cases.jsonl"
    csv_path = OUTPUT_DIR / "pic_mapping.csv"

    for path in (manifest_path, pic_path, chunks_path, risks_path, csv_path):
        assert_true(path.exists(), f"missing output file: {path}")
        assert_true(path.stat().st_size > 0, f"empty output file: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pics = load_jsonl(pic_path)
    chunks = load_jsonl(chunks_path)
    risks = load_jsonl(risks_path)

    assert_true(manifest["manual_count"] == 21, "expected 21 parsed manuals")
    assert_true(len(pics) == manifest["pic_mapping_count"], "pic_mapping count does not match manifest")
    assert_true(len(chunks) == manifest["manual_chunk_count"], "manual_chunks count does not match manifest")
    assert_true(len(risks) == manifest["risk_case_count"], "risk_cases count does not match manifest")

    expected_counts = {
        "EN_SUMMARY": 1783,
        "Manual06": 29,
        "Manual16": 57,
        "Manual18": 102,
        "Manual36": 58,
    }
    by_manual = defaultdict(list)
    for rec in pics:
        by_manual[rec["manual_id"]].append(rec)
        assert_true(rec.get("product"), f"missing product in mapping: {rec}")
        assert_true(isinstance(rec.get("pic_index"), int), f"pic_index must be int: {rec}")
        assert_true(rec.get("mapping_source") in {"human_review", "raw_manual_array"}, f"bad mapping source: {rec}")

    for manual_id, count in expected_counts.items():
        assert_true(len(by_manual[manual_id]) == count, f"{manual_id} mapping count expected {count}, got {len(by_manual[manual_id])}")
        assert_true(
            manifest["manuals"][manual_id]["final_mapping_count"] == count,
            f"{manual_id} manifest final_mapping_count mismatch",
        )

    null_image = [rec for rec in pics if rec.get("image_id") is None]
    assert_true(len(null_image) == 1, f"expected exactly one no-image PIC, got {len(null_image)}")
    assert_true(null_image[0]["manual_id"] == "Manual18" and null_image[0]["pic_index"] == 95, "no-image PIC should be Manual18 PIC95")
    assert_true(null_image[0]["status"] == "extra_pic_no_reliable_image", "Manual18 PIC95 status should preserve human decision")

    manual06_ids = {rec.get("image_id") for rec in by_manual["Manual06"]}
    assert_true("Dish_washer_07" not in manual06_ids, "Dish_washer_07 should be excluded from final dishwasher PIC mapping")
    assert_true("Dish_washer_08" in manual06_ids, "Dish_washer_08 should remain as dishwasher PIC1")

    manual16_ids = [rec.get("image_id") for rec in by_manual["Manual16"]]
    assert_true("Manual16_37" not in manual16_ids, "Manual16_37 should not appear after human confirmation")
    assert_true(Counter(manual16_ids)["Manual16_38"] == 2, "Manual16_38 should be preserved as an intentional duplicate")

    manual18_by_pic = {rec["pic_index"]: rec for rec in by_manual["Manual18"]}
    assert_true(manual18_by_pic[102]["image_id"] == "Manual18_72", "Manual18 PIC102 should map to final circuit diagram")
    assert_true(manual18_by_pic[95]["image_id"] is None, "Manual18 PIC95 should intentionally have no image")

    manual36_by_pic = {rec["pic_index"]: rec for rec in by_manual["Manual36"]}
    assert_true(manual36_by_pic[48]["image_id"] == "Manual36_40", "Manual36 corrected PIC48 should be the inserted DONE/complete image")
    assert_true("DONE" in manual36_by_pic[48]["caption"], "Manual36 PIC48 caption should preserve the DONE/complete meaning")
    assert_true(manual36_by_pic[58]["image_id"] == "Manual36_50", "Manual36 corrected PIC58 should map to final thermostat removal image")

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    assert_true(len(chunk_ids) == len(chunks), "chunk_id values must be unique")
    for manual_id in ("Manual06", "Manual18", "Manual36", "Manual16"):
        pic_chunks = [chunk for chunk in chunks if chunk["manual_id"] == manual_id and chunk["chunk_type"] == "pic_context"]
        assert_true(len(pic_chunks) == len(by_manual[manual_id]), f"{manual_id} pic_context chunk count mismatch")

    risk_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in risks)
    for manual_id in ("Manual06", "Manual18", "Manual36"):
        assert_true(manual_id in risk_text and "raw_count_mismatch" in risk_text, f"missing mismatch risk coverage for {manual_id}")

    print(
        json.dumps(
            {
                "status": "ok",
                "manuals": manifest["manual_count"],
                "pic_mapping": len(pics),
                "chunks": len(chunks),
                "risks": len(risks),
                "checked_manuals": expected_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
