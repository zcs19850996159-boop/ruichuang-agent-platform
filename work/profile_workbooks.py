from __future__ import annotations

import json
import sys
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
END_DIR = ROOT / "work" / "end_unzip" / "end"


def non_empty_rows(ws, limit: int = 12):
    rows = []
    for excel_row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        values = list(row)
        if any(v is not None for v in values):
            while values and values[-1] is None:
                values.pop()
            values = [v[:80] if isinstance(v, str) else v for v in values]
            rows.append({"excel_row": excel_row_idx, "values": values})
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    profiles = []
    for path in sorted(END_DIR.glob("*.xlsx")):
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
        book = {"file": path.name, "sheets": []}
        for ws in wb.worksheets:
            rows = non_empty_rows(ws)
            book["sheets"].append({"sheet": ws.title, "sample_rows": rows})
        wb.close()
        profiles.append(book)
    print(json.dumps(profiles, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
