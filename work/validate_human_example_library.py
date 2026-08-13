from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"
PACK_PATH = ROOT / "outputs" / "rag_agent" / "question_public_human_examples_evidence_packs.jsonl"
REPORT_PATH = ASSET_DIR / "human_example_validation_report.json"
IMAGE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")
CRITICAL_IDS = ["70", "115", "243", "249", "259", "289", "313", "372", "411"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    examples = {str(row["id"]): row for row in load_jsonl(ASSET_DIR / "human_example_library.jsonl")}
    policy_examples = {str(row["id"]): row for row in load_jsonl(ASSET_DIR / "human_policy_examples.jsonl")}
    packs = {str(row["id"]): row for row in load_jsonl(PACK_PATH)}
    sft_rows = load_jsonl(ASSET_DIR / "human_sft_messages.jsonl")

    exact_top = 0
    missing_packs: list[str] = []
    image_mismatches: list[dict[str, Any]] = []
    invalid_image_ids: list[dict[str, Any]] = []
    pic_mismatches: list[dict[str, Any]] = []

    for rid, example in examples.items():
        image_ids = [str(image_id) for image_id in (example.get("image_ids") or [])]
        for image_id in image_ids:
            if not IMAGE_ID_RE.match(image_id):
                invalid_image_ids.append({"id": rid, "image_id": image_id})
        pic_count = str(example.get("answer_sample_plain") or "").count("<PIC>")
        if pic_count != len(image_ids):
            pic_mismatches.append({"id": rid, "pic_count": pic_count, "image_count": len(image_ids), "image_ids": image_ids})
        pack = packs.get(rid)
        if not pack:
            missing_packs.append(rid)
            continue
        top = (pack.get("human_examples") or [{}])[0]
        if top.get("exact_match"):
            exact_top += 1
        if (pack.get("images") or []) != image_ids:
            image_mismatches.append({"id": rid, "pack_images": pack.get("images") or [], "library_images": image_ids})

    sys.path.insert(0, str((ROOT / "work").resolve()))
    import generate_own_answers as gen  # noqa: PLC0415

    policy_source_counts: dict[str, int] = {}
    policy_answer_mismatches: list[dict[str, Any]] = []
    for rid, example in policy_examples.items():
        answer, source = gen.policy_answer_with_source(str(example.get("question") or ""))
        policy_source_counts[source] = policy_source_counts.get(source, 0) + 1
        if answer != example.get("answer_sample_plain"):
            policy_answer_mismatches.append({"id": rid, "source": source})

    critical = {}
    for rid in CRITICAL_IDS:
        example = examples.get(rid, {})
        pack = packs.get(rid, {})
        critical[rid] = {
            "library_images": example.get("image_ids") or [],
            "pack_images": pack.get("images") or [],
            "pic_count": str(example.get("answer_sample_plain") or "").count("<PIC>"),
            "image_count": len(example.get("image_ids") or []),
            "top_exact": bool(((pack.get("human_examples") or [{}])[0]).get("exact_match")) if pack else False,
        }

    report = {
        "manual_examples": len(examples),
        "policy_examples": len(policy_examples),
        "evidence_packs": len(packs),
        "sft_messages": len(sft_rows),
        "manual_exact_top": exact_top,
        "missing_packs": len(missing_packs),
        "image_mismatches": len(image_mismatches),
        "invalid_image_ids": len(invalid_image_ids),
        "pic_image_mismatches": len(pic_mismatches),
        "image_without_pic_examples": sum(1 for row in pic_mismatches if row["image_count"] > 0 and row["pic_count"] == 0),
        "policy_source_counts": policy_source_counts,
        "policy_answer_mismatches": len(policy_answer_mismatches),
        "critical_ids": critical,
        "samples": {
            "image_mismatches": image_mismatches[:10],
            "invalid_image_ids": invalid_image_ids[:10],
            "pic_mismatches": pic_mismatches[:20],
            "policy_answer_mismatches": policy_answer_mismatches[:10],
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
