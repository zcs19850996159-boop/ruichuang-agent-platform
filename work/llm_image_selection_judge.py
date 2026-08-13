from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from image_selector import ImageSelector


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in load_jsonl(path)}


def call_deepseek(messages: list[dict[str, str]], *, timeout: float = 45.0) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.0")),
        "max_tokens": int(os.environ.get("DEEPSEEK_MAX_TOKENS", "700")),
        "response_format": {"type": "json_object"},
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def parse_ids(text: str, allowed: set[str]) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return []
        data = json.loads(match.group(0))
    raw = data.get("image_ids") or data.get("images") or []
    if isinstance(raw, str):
        raw = re.findall(r"(?:Manual\d+|[A-Za-z_]+)_[A-Za-z0-9]+", raw)
    out: list[str] = []
    for image_id in raw:
        image_id = str(image_id).strip()
        if image_id in allowed and image_id not in out:
            out.append(image_id)
    return out


def score_f1(pred: list[str], gold: list[str]) -> float:
    ps = set(pred)
    gs = set(gold)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    rec = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def proposal_text(selector: ImageSelector, name: str, image_ids: list[str]) -> list[str]:
    lines = [f"{name}: count={len(image_ids)} ids={image_ids}"]
    for image_id in image_ids:
        rec = selector.image_records.get(str(image_id))
        text = compact(rec.text if rec else "")[:240]
        lines.append(f"  - {image_id}: {text}")
    return lines


def render_prompt(selector: ImageSelector, row: dict[str, Any], proposals: dict[str, list[str]]) -> tuple[list[dict[str, str]], set[str]]:
    allowed: set[str] = set()
    for ids in proposals.values():
        allowed.update(str(image_id) for image_id in ids)
    lines = [
        "You are the final judge for manual figure selection.",
        "Return JSON only: {\"image_ids\": [\"...\"]}.",
        "Choose only from IDs already present in the proposals. You may choose one proposal unchanged, choose a subset, or merge proposals.",
        "If a calibrated_meta_selector proposal is present, treat it as the default calibrated choice. Keep it unless another proposal is clearly more directly supported by the question and figure captions.",
        "Prefer leaving a coherent proposal unchanged over making small speculative edits.",
        "Select figures that directly support the customer answer; remove weak neighboring figures.",
        "Prefer exact heading/topic match, correct subsection, and contiguous step sequences when the question asks for steps.",
        "For safety questions, prefer the requested safety subsection and avoid unrelated task-specific warnings.",
        "For component/package questions, prefer overview/package/parts figures instead of later use or troubleshooting figures.",
        "For definition questions, include adjacent figures only when they complete the same small subsection.",
        "Keep final IDs in manual/PIC order.",
        "",
        f"Manual: {row.get('manual')}",
        f"Question: {row.get('question')}",
        "",
        "Proposals:",
    ]
    for name, ids in proposals.items():
        lines.extend(proposal_text(selector, name, ids))
    return (
        [
            {"role": "system", "content": "You are a strict evidence-image judge. Output valid JSON only."},
            {"role": "user", "content": "\n".join(lines)},
        ],
        allowed,
    )


def run_one(selector: ImageSelector, row_id: str, rows_by_variant: dict[str, dict[str, dict[str, Any]]], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    base_row = rows_by_variant["score"][row_id]
    proposals = {
        "score_order_model": [str(x) for x in rows_by_variant["score"][row_id].get("pred") or []],
        "manual_order_model": [str(x) for x in rows_by_variant["manual"][row_id].get("pred") or []],
        "early_prompt_model": [str(x) for x in rows_by_variant["v1"][row_id].get("pred") or []],
        "heuristic_model": [str(x) for x in base_row.get("base_pred") or []],
    }
    if rows_by_variant.get("meta") and row_id in rows_by_variant["meta"]:
        proposals = {
            "calibrated_meta_selector": [str(x) for x in rows_by_variant["meta"][row_id].get("pred") or []],
            **proposals,
        }
    unique_sets = {tuple(ids) for ids in proposals.values()}
    raw = ""
    if len(unique_sets) == 1:
        pred = next(iter(proposals.values()))
    else:
        prompt, allowed = render_prompt(selector, base_row, proposals)
        raw = call_deepseek(prompt, timeout=timeout)
        pred = parse_ids(raw, allowed)
        if not pred and any(proposals.values()):
            pred = proposals["score_order_model"]
    gold = [str(x) for x in base_row.get("gold") or []]
    f1 = score_f1(pred, gold)
    return {
        "id": row_id,
        "manual": base_row.get("manual"),
        "question": base_row.get("question"),
        "gold": gold,
        "pred": pred,
        "proposals": proposals,
        "raw": raw,
        "f1": round(f1, 6),
        "gold_n": len(gold),
        "pred_n": len(pred),
        "exact": pred == gold,
        "seteq": set(pred) == set(gold),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_rows = [r for r in rows if r["gold_n"] > 0]
    no_rows = [r for r in rows if r["gold_n"] == 0]
    return {
        "rows": len(rows),
        "exact_order": sum(1 for r in rows if r["exact"]),
        "set_equal": sum(1 for r in rows if r["seteq"]),
        "avg_f1": sum(float(r["f1"]) for r in rows) / max(1, len(rows)),
        "image_rows": len(image_rows),
        "image_avg_f1": sum(float(r["f1"]) for r in image_rows) / max(1, len(image_rows)),
        "gold_no_image": len(no_rows),
        "pred_no_image": sum(1 for r in rows if r["pred_n"] == 0),
        "no_image_correct": sum(1 for r in no_rows if r["pred_n"] == 0),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", default="outputs/rag_agent/llm_selector_full_v4_score_k50.jsonl")
    parser.add_argument("--manual", default="outputs/rag_agent/llm_selector_full_v4_manual_k50.jsonl")
    parser.add_argument("--v1", default="outputs/rag_agent/llm_selector_full_v1.jsonl")
    parser.add_argument("--meta", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    rows_by_variant = {
        "score": by_id(ROOT / args.score),
        "manual": by_id(ROOT / args.manual),
        "v1": by_id(ROOT / args.v1),
    }
    if args.meta:
        rows_by_variant["meta"] = by_id(ROOT / args.meta)
    ids = sorted(rows_by_variant["score"], key=lambda x: int(x) if str(x).isdigit() else str(x))
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        ids = [row_id for row_id in ids if row_id in wanted]

    selector = ImageSelector("v59")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, selector, row_id, rows_by_variant, args.timeout): row_id
            for row_id in ids
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[done] {row['id']} f1={row['f1']} pred={row['pred_n']} gold={row['gold_n']} {row['elapsed_ms']}ms", flush=True)
    rows.sort(key=lambda r: int(r["id"]) if str(r["id"]).isdigit() else str(r["id"]))
    write_jsonl(ROOT / args.output, rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
