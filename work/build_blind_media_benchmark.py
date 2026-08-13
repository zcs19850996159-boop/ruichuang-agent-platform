from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = ("crop", "rotate", "compress_blur", "glare", "occlusion", "affine")


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def excluded_image_ids(rule_path: Path) -> set[str]:
    if not rule_path.is_file():
        return set()
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for rule in payload.get("rules") or []:
        result.update(str(item) for item in (rule.get("image_ids") or []) if item)
    return result


def category(row: dict[str, Any]) -> str:
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    if structured.get("error_codes"):
        return "error_code"
    if structured.get("buttons") or structured.get("screen_state"):
        return "control_panel"
    if structured.get("components"):
        return "component"
    if structured.get("ocr_text"):
        return "ocr"
    return "procedure"


def question_for(kind: str) -> str:
    return {
        "error_code": "请识别图片中的错误码或告警状态，并根据对应产品手册说明含义和处理方法。",
        "control_panel": "请识别图片中的控制区域、按钮或屏幕状态，并根据产品手册说明如何操作。",
        "component": "图片中是什么产品部件？请根据对应手册说明它的用途或正确使用方法。",
        "ocr": "请结合图片中可见文字和对应产品手册，说明图片展示的内容及正确操作方法。",
        "procedure": "请识别图片展示的产品步骤或状态，并根据对应手册给出有证据的操作说明。",
    }[kind]


def rubric_terms(row: dict[str, Any]) -> list[str]:
    structured = row.get("structured") if isinstance(row.get("structured"), dict) else {}
    values: list[str] = []
    for key in ("error_codes", "visual_tags", "components", "buttons"):
        for item in structured.get(key) or []:
            if isinstance(item, dict):
                value = item.get("name") or item.get("code") or item.get("description")
            else:
                value = item
            if value:
                values.append(str(value).strip())
    for key in ("screen_state", "ocr_text", "caption"):
        if structured.get(key):
            values.append(str(structured[key]).strip())
    return list(dict.fromkeys(value for value in values if len(value) >= 2))[:12]


def realistic_variant(image: Image.Image, transform_name: str, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if transform_name == "crop":
        margin_x = max(1, int(width * rng.uniform(0.04, 0.10)))
        margin_y = max(1, int(height * rng.uniform(0.04, 0.10)))
        return image.crop((margin_x, margin_y, width - margin_x, height - margin_y)).resize(
            (width, height), Image.Resampling.LANCZOS
        )
    if transform_name == "rotate":
        return image.rotate(rng.uniform(-4.0, 4.0), resample=Image.Resampling.BICUBIC, fillcolor="white")
    if transform_name == "compress_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.45, 0.95)))
    if transform_name == "glare":
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        offset = rng.randint(-max(1, width // 4), max(1, width // 4))
        draw.polygon(
            [
                (offset, 0),
                (offset + max(20, width // 7), 0),
                (offset + width // 2, height),
                (offset + width // 3, height),
            ],
            fill=(255, 255, 255, 58),
        )
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    if transform_name == "occlusion":
        result = image.copy()
        draw = ImageDraw.Draw(result, "RGBA")
        box_width = max(10, int(width * 0.12))
        box_height = max(10, int(height * 0.10))
        left = rng.randint(0, max(0, width - box_width))
        top = rng.randint(0, max(0, height - box_height))
        draw.rounded_rectangle(
            (left, top, left + box_width, top + box_height),
            radius=max(2, min(box_width, box_height) // 6),
            fill=(45, 45, 45, 115),
        )
        return result
    shear = rng.uniform(-0.06, 0.06)
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, shear, -shear * height / 2, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge",
        default=str(ROOT / "outputs" / "rag_assets" / "image_knowledge_auto.jsonl"),
    )
    parser.add_argument(
        "--rules",
        default=str(ROOT / "assets" / "visual_grounding_rules.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "blind_media_benchmark_v1"),
    )
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-per-manual", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    excluded = excluded_image_ids(Path(args.rules))
    candidates: list[dict[str, Any]] = []
    rejected = defaultdict(int)
    for row in read_jsonl(Path(args.knowledge)):
        image_id = str(row.get("image_id") or "")
        path = Path(str(row.get("path") or row.get("source") or ""))
        if not image_id or image_id in excluded:
            rejected["official_rule_image"] += 1
            continue
        if row.get("status") != "ok" or not path.is_file():
            rejected["unusable"] += 1
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            rejected["decode_failed"] += 1
            continue
        if min(width, height) < 120 or width * height < 30000:
            rejected["too_small"] += 1
            continue
        if not row.get("manual_id") or not row.get("nearby_text"):
            rejected["missing_evidence"] += 1
            continue
        row = dict(row)
        row["_category"] = category(row)
        row["_stable"] = stable_key(args.seed, image_id)
        candidates.append(row)

    selected: list[dict[str, Any]] = []
    manual_counts: dict[str, int] = defaultdict(int)
    selected_ids: set[str] = set()
    quotas = {
        "error_code": max(1, round(args.count * 0.08)),
        "control_panel": max(1, round(args.count * 0.20)),
        "ocr": max(1, round(args.count * 0.17)),
        "component": max(1, round(args.count * 0.33)),
    }
    quotas["procedure"] = max(1, args.count - sum(quotas.values()))
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_category[row["_category"]].append(row)
    for rows in by_category.values():
        rows.sort(key=lambda item: (item["_stable"], item["image_id"]))

    def add_from(rows: list[dict[str, Any]], wanted: int) -> None:
        for row in sorted(
            rows,
            key=lambda item: (
                manual_counts[str(item["manual_id"])],
                item["_stable"],
                item["image_id"],
            ),
        ):
            if wanted <= 0 or len(selected) >= args.count:
                return
            image_id = str(row["image_id"])
            manual_id = str(row["manual_id"])
            if image_id in selected_ids or manual_counts[manual_id] >= args.max_per_manual:
                continue
            selected.append(row)
            selected_ids.add(image_id)
            manual_counts[manual_id] += 1
            wanted -= 1

    for kind in ("error_code", "control_panel", "ocr", "component", "procedure"):
        add_from(by_category.get(kind, []), quotas[kind])
    if len(selected) < args.count:
        add_from(candidates, args.count - len(selected))

    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        case_id = f"blind-img-{index:03d}"
        transform_name = TRANSFORMS[(index - 1) % len(TRANSFORMS)]
        rng = random.Random(f"{args.seed}:{row['image_id']}:{transform_name}")
        with Image.open(str(row.get("path") or row.get("source"))) as source:
            variant = realistic_variant(source, transform_name, rng)
        output_path = images_dir / f"{case_id}.jpg"
        variant.save(
            output_path,
            format="JPEG",
            quality=55 if transform_name == "compress_blur" else 82,
            optimize=True,
        )
        manifest_rows.append(
            {
                "case_id": case_id,
                "split": "blind_v1_candidate",
                "question": question_for(row["_category"]),
                "category": row["_category"],
                "input_path": str(output_path),
                "transform": transform_name,
                "expected": {
                    "manual_id": row.get("manual_id"),
                    "product": row.get("product"),
                    "image_id": row.get("image_id"),
                    "section_title": row.get("section_title"),
                    "section_path": row.get("section_path"),
                    "source_page": row.get("source_page"),
                    "evidence": str(row.get("nearby_text") or "")[:1200],
                    "rubric_terms": rubric_terms(row),
                },
                "source": {
                    "source_hash": row.get("source_hash"),
                    "chunk_version": row.get("chunk_version"),
                },
                "review_status": "needs_human_review",
                "rule_bypass_required": True,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "seed": args.seed,
        "requested": args.count,
        "created": len(manifest_rows),
        "candidate_pool": len(candidates),
        "excluded_image_ids": len(excluded),
        "manual_count": len({row["expected"]["manual_id"] for row in manifest_rows}),
        "category_counts": dict(
            sorted(
                {
                    kind: sum(row["category"] == kind for row in manifest_rows)
                    for kind in {row["category"] for row in manifest_rows}
                }.items()
            )
        ),
        "transform_counts": {
            name: sum(row["transform"] == name for row in manifest_rows)
            for name in TRANSFORMS
        },
        "rejected": dict(rejected),
        "review_status": "needs_human_review",
        "warning": "Do not report candidate results as final blind accuracy before human review.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
