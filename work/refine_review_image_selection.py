from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"


IMAGE_ID_PATTERN = re.compile(
    r"(?:Manual\d+|Camera|drill\d*|jetski|Security_Camera|air_conditioner|exercise_bikes|fitness_trackers|oven|fax|generator|Dish_washer|Blower|earphones|eReader|air_fryer|multi-use_pressure_cooker_and_air_fryer)_[A-Za-z0-9]+"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_caption_index() -> dict[str, str]:
    captions: dict[str, str] = {}
    for rec in load_jsonl(ASSET_DIR / "pic_mapping.jsonl"):
        image_id = str(rec.get("image_id") or "")
        if not image_id or image_id in captions:
            continue
        text = compact(
            " ".join(
                str(rec.get(key) or "")
                for key in ("caption", "section", "before_context", "after_context", "notes")
            )
        )
        if text:
            captions[image_id] = text
    for rec in load_jsonl(ASSET_DIR / "english_pic_captions.jsonl"):
        image_id = str(rec.get("image_id") or "")
        if not image_id or image_id in captions:
            continue
        text = compact(
            " ".join(
                str(rec.get(key) or "")
                for key in ("caption_en", "nearest_section", "section_path", "notes")
            )
        )
        if text:
            captions[image_id] = text
    for rec in load_jsonl(ASSET_DIR / "original_manual_reverse_checks.jsonl"):
        image_id = str(rec.get("image_id") or "")
        if not image_id or image_id in captions:
            continue
        text = compact(" ".join(str(rec.get(key) or "") for key in ("caption_cn", "conclusion", "action")))
        if text:
            captions[image_id] = text
    return captions


def preview_caption_map(preview: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(preview or "").splitlines():
        if ":" not in line:
            continue
        image_id, text = line.split(":", 1)
        image_id = image_id.strip()
        if IMAGE_ID_PATTERN.fullmatch(image_id):
            out[image_id] = compact(text)
    return out


def parse_json_response(text: str, candidates: list[str]) -> tuple[list[str], str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except Exception:
        ids = [image_id for image_id in IMAGE_ID_PATTERN.findall(cleaned) if image_id in candidates]
        return list(dict.fromkeys(ids)), cleaned[:400]
    selected = data.get("selected") or data.get("selected_images") or []
    if not isinstance(selected, list):
        selected = []
    selected = [str(image_id) for image_id in selected if str(image_id) in candidates]
    return list(dict.fromkeys(selected)), compact(str(data.get("reason") or ""))[:600]


def call_deepseek(row: dict[str, Any], candidates: list[dict[str, str]], model: str, timeout: float) -> tuple[list[str], str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    candidate_ids = [item["image_id"] for item in candidates]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are selecting image IDs for a product manual answer. "
                    "Choose only from the candidate image IDs. Return JSON only. "
                    "Every selected image must be directly explained by a sentence or step in the final answer. "
                    "Prefer a minimal set. Keep multiple images only for true ordered procedures where each image maps to a distinct key step. "
                    "For a single component, button, table, definition, or parameter question, keep only the directly relevant image. "
                    "If all candidates are weak, cross-manual, or unrelated, return an empty list."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question_id": row["id"],
                        "question": row["question"],
                        "manual_id": row["manual_id"],
                        "product": row["product"],
                        "risk": row["risk"],
                        "review_action": row["action"],
                        "review_issue": row["issue"],
                        "review_feedback": row["model_feedback"],
                        "candidates": candidates,
                        "output_schema": {"selected": ["image_id"], "reason": "short reason"},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 700,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_json_response(data["choices"][0]["message"]["content"], candidate_ids)


def needs_llm(row: dict[str, Any]) -> bool:
    if row.get("suggested_images"):
        return False
    if row.get("action") == "保留":
        return False
    return bool(row.get("current_images"))


def build_one(row: dict[str, Any], captions: dict[str, str], model: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    current = [str(image_id) for image_id in row.get("current_images") or [] if str(image_id).strip()]
    preview = preview_caption_map(row.get("caption_preview") or "")
    if row.get("suggested_images"):
        selected = [str(image_id) for image_id in row["suggested_images"] if str(image_id).strip()]
        source = "human_suggested_images"
        reason = "Used the explicit suggested image array from the review workbook."
    elif row.get("action") == "保留":
        selected = current
        source = "human_review_keep"
        reason = "Review workbook marked this image set as keep."
    elif not current:
        selected = []
        source = "empty_current"
        reason = "No current candidate images."
    else:
        candidates = [
            {
                "image_id": image_id,
                "caption": (preview.get(image_id) or captions.get(image_id) or "")[:700],
            }
            for image_id in current
        ]
        try:
            selected, reason = call_deepseek(row, candidates, model, timeout)
            source = "deepseek_review_prune"
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            selected = current
            source = "fallback_current_on_error"
            reason = f"{type(exc).__name__}: {exc}"
    return {
        "id": row["id"],
        "manual_id": row["manual_id"],
        "question": row["question"],
        "review_action": row["action"],
        "review_risk": row["risk"],
        "current_images": current,
        "selected_images": selected,
        "source": source,
        "reason": reason,
        "review_issue": row.get("issue", ""),
        "model_feedback": row.get("model_feedback", ""),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default=str(ASSET_DIR / "question_review_rules.jsonl"))
    parser.add_argument("--output", default=str(ASSET_DIR / "question_review_image_overrides.jsonl"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("DEEPSEEK_TIMEOUT", "45")))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.rules))
    captions = build_caption_index()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(build_one, row, captions, args.model, args.timeout): row
            for row in rows
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            delta = len(result["selected_images"]) - len(result["current_images"])
            print(
                f"[{result['source']}] {result['id']} {len(result['current_images'])}->{len(result['selected_images'])} delta={delta} {result['elapsed_ms']}ms",
                flush=True,
            )
    results.sort(key=lambda item: int(item["id"]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"finished {len(results)} rows")


if __name__ == "__main__":
    main()
