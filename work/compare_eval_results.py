from __future__ import annotations

import argparse
import csv
import json


def load_rows(path: str) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as source:
        return {str(row.get("id") or ""): row for row in csv.DictReader(source)}


def canonical_json(value: str) -> object:
    try:
        return json.loads(value or "[]")
    except json.JSONDecodeError:
        return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline = load_rows(args.baseline)
    candidate = load_rows(args.candidate)
    ids = sorted(set(baseline) | set(candidate), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    fields = (
        "http_status",
        "api_code",
        "route_type",
        "manual_id",
        "constraint_pass",
        "pic_count",
        "image_count",
        "pic_match_images",
        "image_exact_set",
        "image_exact_order",
        "remote_media_detected",
        "remote_media_usable",
        "vision_used",
        "retrieval_mode",
        "retrieval_decision",
        "refusal_type",
        "images",
        "answer",
    )
    differences: dict[str, list[str]] = {field: [] for field in fields}
    missing: list[str] = []
    for row_id in ids:
        left = baseline.get(row_id)
        right = candidate.get(row_id)
        if left is None or right is None:
            missing.append(row_id)
            continue
        for field in fields:
            left_value: object = left.get(field, "")
            right_value: object = right.get(field, "")
            if field == "images":
                left_value = canonical_json(str(left_value))
                right_value = canonical_json(str(right_value))
            if left_value != right_value:
                differences[field].append(row_id)
    summary = {
        "rows": len(ids),
        "missing_rows": missing,
        "difference_counts": {field: len(row_ids) for field, row_ids in differences.items()},
        "difference_ids": {field: row_ids for field, row_ids in differences.items() if row_ids},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    structural_fields = fields[:-1]
    return 1 if missing or any(differences[field] for field in structural_fields) else 0


if __name__ == "__main__":
    raise SystemExit(main())
