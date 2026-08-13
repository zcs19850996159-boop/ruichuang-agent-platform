from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric",
        default=str(ROOT / "assets" / "blind_pilot12_reviewed_v1.json"),
    )
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    for result_path in args.results:
        results.update(read_jsonl(Path(result_path)))
    scored: list[dict[str, Any]] = []
    for case in rubric.get("cases") or []:
        result = results.get(case["case_id"], {})
        answer = str(result.get("answer") or "")
        images = [str(item) for item in (result.get("actual_images") or result.get("images") or [])]
        group_hits = [
            any(str(term).lower() in answer.lower() for term in group)
            for group in (case.get("answer_must_contain_groups") or [])
        ]
        forbidden_hits = [
            term
            for term in (case.get("answer_must_not_contain") or [])
            if str(term).lower() in answer.lower()
        ]
        acceptable = [str(item) for item in (case.get("acceptable_images") or [])]
        image_supported = not acceptable or any(item in acceptable for item in images)
        row = {
            "case_id": case["case_id"],
            "review_status": case["review_status"],
            "manual_match": str(result.get("actual_manual_id") or "")
            == str(case.get("manual_id") or ""),
            "answer_groups_pass": bool(group_hits) and all(group_hits),
            "answer_group_hits": group_hits,
            "forbidden_pass": not forbidden_hits,
            "forbidden_hits": forbidden_hits,
            "image_supported": image_supported,
            "actual_manual_id": result.get("actual_manual_id"),
            "actual_images": images,
            "human_approval_required": True,
        }
        row["automatic_pass"] = all(
            bool(row[key])
            for key in ("manual_match", "answer_groups_pass", "forbidden_pass", "image_supported")
        )
        scored.append(row)

    summary = {
        "total": len(scored),
        "automatic_pass": sum(bool(row["automatic_pass"]) for row in scored),
        "manual_match": sum(bool(row["manual_match"]) for row in scored),
        "answer_groups_pass": sum(bool(row["answer_groups_pass"]) for row in scored),
        "forbidden_pass": sum(bool(row["forbidden_pass"]) for row in scored),
        "image_supported": sum(bool(row["image_supported"]) for row in scored),
        "needs_human_confirmation": sum(
            row["review_status"] == "needs_human_confirmation" for row in scored
        ),
        "human_approval_required": True,
    }
    payload = {"summary": summary, "cases": scored}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
