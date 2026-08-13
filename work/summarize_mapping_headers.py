from __future__ import annotations

import json
import sys
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
END_DIR = ROOT / "work" / "end_unzip" / "end"
TOKENS = ("pic", "PIC", "图片编号", "image_id", "pic_index", "caption", "校正后图片编号")


def clean(row):
    values = list(row)
    while values and values[-1] is None:
        values.pop()
    return values


def main() -> None:
    items = []
    for path in sorted(END_DIR.glob("*.xlsx")):
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
        for ws in wb.worksheets:
            for idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                values = clean(row)
                text = " ".join(str(v) for v in values if v is not None)
                if any(tok in text for tok in TOKENS):
                    items.append(
                        {
                            "file": path.name,
                            "sheet": ws.title,
                            "header_row": idx,
                            "values": values,
                        }
                    )
                    break
        wb.close()
    print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
