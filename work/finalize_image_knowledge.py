from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    manifest = load_jsonl(Path(args.manifest))
    expected = {str(row.get("image_id") or ""): row for row in manifest if row.get("image_id")}
    latest_ok: dict[str, dict[str, Any]] = {}
    latest_error: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    for shard in args.shards:
        for row in load_jsonl(Path(shard)):
            raw_rows += 1
            image_id = str(row.get("image_id") or "")
            if not image_id:
                continue
            if row.get("status") == "ok":
                latest_ok[image_id] = row
            else:
                latest_error[image_id] = row

    ordered = [latest_ok[image_id] for image_id in expected if image_id in latest_ok]
    missing = [image_id for image_id in expected if image_id not in latest_ok]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
    os.replace(temp, output)
    summary = {
        "version": "image-knowledge-auto-v1",
        "created_at": int(time.time()),
        "manifest_count": len(expected),
        "raw_output_rows": raw_rows,
        "success_count": len(ordered),
        "missing_count": len(missing),
        "coverage": round(len(ordered) / max(1, len(expected)), 6),
        "missing_image_ids": missing[:100],
        "error_samples": [
            {"image_id": image_id, "error": latest_error[image_id].get("error")}
            for image_id in missing[:100]
            if image_id in latest_error
        ],
    }
    summary_path = Path(args.summary)
    summary_temp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(summary_temp, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
