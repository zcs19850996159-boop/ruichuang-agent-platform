from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * value) - 1))
    return round(ordered[index], 2)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows if row.get("elapsed_ms") is not None]
    first = [float(row["first_delta_ms"]) for row in rows if row.get("first_delta_ms") is not None]
    return {
        "count": len(rows),
        "elapsed_p50_ms": round(statistics.median(elapsed), 2) if elapsed else None,
        "elapsed_p95_ms": percentile(elapsed, 0.95),
        "first_delta_p50_ms": round(statistics.median(first), 2) if first else None,
        "first_delta_p95_ms": percentile(first, 0.95),
        "error_count": sum(int(row.get("status", 200)) >= 400 for row in rows),
        "timeout_count": sum(str(row.get("outcome") or "") == "timeout" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default="historical")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.metrics).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [row for row in rows if int(row.get("status", 200)) == 200]
    report = {
        "environment": {
            "platform": platform.platform(),
            "scenario": args.scenario,
            "concurrency": args.concurrency,
            "warm": args.warm,
        },
        "text": summarize([row for row in successful if not row.get("input_image_count")]),
        "image": summarize([row for row in successful if row.get("input_image_count")]),
        "all": summarize(rows),
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

