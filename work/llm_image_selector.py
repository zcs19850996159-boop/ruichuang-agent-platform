from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from image_selector import ImageSelector, expand_query_text, token_counter


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
        "max_tokens": int(os.environ.get("DEEPSEEK_MAX_TOKENS", "900")),
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


def parse_json_ids(text: str, allowed: set[str]) -> list[str]:
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
    ids: list[str] = []
    for image_id in raw:
        image_id = str(image_id).strip()
        if image_id in allowed and image_id not in ids:
            ids.append(image_id)
    return ids


def build_candidates(selector: ImageSelector, row_id: str, question: str, manual_id: str, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = selector.routes.get(row_id)
    similar = selector._similar_examples(row_id, question, manual_id, leave_one_out=True)
    q_tokens = token_counter(expand_query_text(" ".join([question, route.intent_type if route else ""])))
    candidates = selector._score_candidates(row_id, question, manual_id, q_tokens, similar, leave_one_out=True)
    base = selector.select(row_id, question, manual_id, leave_one_out=True, debug=False)
    return candidates[:k], base


def render_prompt(
    question: str,
    manual_id: str,
    candidates: list[dict[str, Any]],
    base_images: list[str],
    similar_examples: list[dict[str, Any]],
    candidate_order: str,
) -> list[dict[str, str]]:
    score_rank = {str(c["image_id"]): idx + 1 for idx, c in enumerate(candidates)}
    if candidate_order == "manual":
        ordered_candidates = sorted(
            candidates,
            key=lambda c: (
                int(c.get("pic_index") if c.get("pic_index") is not None else 10**9),
                str(c.get("image_id") or ""),
            ),
        )
        candidate_heading = "Candidate figures in manual/PIC order:"
    else:
        ordered_candidates = candidates
        candidate_heading = "Candidate figures in retrieval-score order:"
    lines = [
        "You are selecting figures from a product manual for a customer-service answer.",
        "Return JSON only: {\"image_ids\": [\"...\"]}.",
        "Choose only from the candidate IDs. Select every figure that directly supports the answer; exclude weak neighboring figures.",
        "If the question is text-only, policy-only, warranty-only, or the candidate figures do not directly support it, return an empty array.",
        "For step-by-step installation/replacement/removal questions, include the full directly relevant step sequence, not just one overview figure.",
        "For diagrams, indicator meanings, component/anatomy views, or mode explanations, include all figures needed to explain the requested items.",
        "For component/package/what-should-I-have questions, prefer product overview, labeled parts, package contents, or initial component figures; avoid later usage/charging/troubleshooting figures.",
        "For safety questions, use the figures in the requested safety subsection. If the question asks general safety/symbols, prefer the initial safety-symbol figures; do not add later task-specific warnings unless named.",
        "For questions that specify first/last N steps, choose figures whose captions or nearby text match those exact step numbers and subsection, not other procedures.",
        "For a definition question like 'what is X', include the figure at the X heading plus immediately adjacent figures that complete that same small subsection.",
        "Retrieval scores are noisy hints only. Prefer exact heading/topic match and contiguous manual sequence over a higher score.",
        "Keep the final order in manual/PIC order, not by score.",
        "",
        f"Manual: {manual_id}",
        f"Question: {question}",
        f"Heuristic initial images: {base_images}",
    ]
    if similar_examples:
        lines.append("\nNon-exact similar training examples:")
        for example in similar_examples[:4]:
            answer_sample = re.sub(r"\s+", " ", str(example.get("answer_sample") or ""))[:260]
            lines.append(
                f"- q={example.get('question','')[:150]} pic_count={example.get('pic_count')} images={example.get('images') or example.get('image_ids') or []} answer_sample={answer_sample}"
            )
    lines.append("\n" + candidate_heading)
    for idx, cand in enumerate(ordered_candidates, 1):
        text = re.sub(r"\s+", " ", str(cand.get("text") or ""))[:520]
        lines.append(
            f"{idx}. id={cand['image_id']} pic_index={cand.get('pic_index')} retrieval_rank={score_rank.get(str(cand['image_id']))} score={cand.get('score')} text={text}"
        )
    return [
        {
            "role": "system",
            "content": "You are a strict manual figure selector. Output valid JSON only.",
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_one(selector: ImageSelector, example: Any, candidate_k: int, timeout: float, candidate_order: str) -> dict[str, Any]:
    started = time.perf_counter()
    route = selector.routes.get(example.row_id)
    question = route.question if route else example.question
    base = selector.select(example.row_id, question, example.manual_id, leave_one_out=True, debug=False)
    if not example.image_ids and not base.get("image_ids"):
        pred: list[str] = []
        raw = ""
        candidates: list[dict[str, Any]] = []
    else:
        candidates, base = build_candidates(selector, example.row_id, question, example.manual_id, candidate_k)
        allowed = {str(c["image_id"]) for c in candidates}
        prompt = render_prompt(
            question,
            example.manual_id,
            candidates,
            [str(x) for x in (base.get("image_ids") or [])],
            base.get("similar_examples") or [],
            candidate_order,
        )
        raw = call_deepseek(prompt, timeout=timeout)
        pred = parse_json_ids(raw, allowed)
        if not pred and example.image_ids:
            pred = [str(x) for x in (base.get("image_ids") or [])]
    gold = example.image_ids
    gs = set(gold)
    ps = set(pred)
    tp = len(gs & ps)
    prec = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    rec = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "id": example.row_id,
        "manual": example.manual_id,
        "question": question,
        "gold": gold,
        "pred": pred,
        "base_pred": base.get("image_ids") or [],
        "candidate_ids": [c["image_id"] for c in candidates],
        "raw": raw,
        "exact": pred == gold,
        "seteq": ps == gs,
        "gold_n": len(gold),
        "pred_n": len(pred),
        "f1": round(f1, 6),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        "missed_needed_images": sum(1 for r in image_rows if r["pred_n"] == 0),
        "false_images_for_no_image": sum(1 for r in no_rows if r["pred_n"] > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", help="Comma-separated IDs to run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--candidate-order", choices=["score", "manual"], default="score")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    selector = ImageSelector("v59")
    examples = selector.teacher_examples
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        examples = [ex for ex in examples if ex.row_id in wanted]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, selector, ex, args.candidate_k, args.timeout, args.candidate_order): ex for ex in examples}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[done] {row['id']} f1={row['f1']} pred={row['pred_n']} gold={row['gold_n']} {row['elapsed_ms']}ms", flush=True)
    rows.sort(key=lambda r: int(r["id"]) if str(r["id"]).isdigit() else str(r["id"]))
    write_jsonl(Path(args.output), rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
