from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    text = compact(text).lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    out: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 3:
            out.extend(token[i : i + 2] for i in range(len(token) - 1))
            out.extend(token[i : i + 3] for i in range(len(token) - 2))
        else:
            out.append(token)
    stop = {
        "how", "what", "when", "where", "can", "could", "should", "would", "the", "and", "or", "for", "with",
        "using", "use", "about", "manual", "answer", "image", "question", "如何", "什么", "哪些", "手册", "使用",
    }
    return [tok for tok in out if tok not in stop and len(tok) > 1]


def image_key(image_id: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)$", image_id or "")
    return (re.sub(r"_.*$", "", image_id or ""), int(match.group(1)) if match else 10**9, image_id or "")


def record_image_id(rec: dict[str, Any]) -> str:
    return str(rec.get("image_id") or "")


def record_manual_id(rec: dict[str, Any]) -> str:
    return str(rec.get("manual_id") or "")


def record_text(rec: dict[str, Any]) -> str:
    return compact(
        " ".join(
            str(rec.get(key) or "")
            for key in (
                "image_id",
                "caption",
                "caption_en",
                "nearest_section",
                "section",
                "section_path",
                "before_context",
                "after_context",
            )
        )
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def deepseek_chat(messages: list[dict[str, str]], model: str, timeout: float, max_tokens: int = 900) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail}") from exc


class ImageRepairer:
    def __init__(self, packs_path: Path, results_path: Path, overrides_path: Path) -> None:
        self.packs = {str(row["id"]): row for row in load_jsonl(packs_path)}
        self.results = {str(row["id"]): row for row in load_jsonl(results_path)}
        self.overrides = {str(row["id"]): row for row in load_jsonl(overrides_path)}
        self.original_override_rows = load_jsonl(overrides_path)
        records: list[dict[str, Any]] = []
        for rec in load_jsonl(ASSET_DIR / "pic_mapping.jsonl"):
            if rec.get("image_id"):
                records.append(rec)
        for rec in load_jsonl(ASSET_DIR / "english_pic_captions.jsonl"):
            if rec.get("image_id"):
                records.append(
                    {
                        "image_id": rec.get("image_id"),
                        "manual_id": rec.get("manual_id"),
                        "caption": rec.get("caption_en"),
                        "section": rec.get("nearest_section") or rec.get("section_path"),
                        "before_context": "",
                        "after_context": rec.get("section_path"),
                    }
                )
        self.records_by_manual: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.records_by_id: dict[str, dict[str, Any]] = {}
        for rec in records:
            image_id = record_image_id(rec)
            manual_id = record_manual_id(rec)
            if image_id:
                self.records_by_id.setdefault(image_id, rec)
            if manual_id and image_id:
                self.records_by_manual[manual_id].append(rec)

    def failing_ids(self, limit: int | None = None) -> list[str]:
        ids = [rid for rid, row in self.results.items() if row.get("constraint_pass") is False]
        ids.sort(key=lambda rid: int(rid) if rid.isdigit() else rid)
        return ids[:limit] if limit else ids

    def candidate_records(self, pack: dict[str, Any], result: dict[str, Any], limit: int = 36) -> list[dict[str, Any]]:
        constraints = pack.get("answer_constraints") or {}
        manual_id = str(constraints.get("manual_id") or (pack.get("route") or {}).get("manual_id") or "")
        query = " ".join(
            [
                pack.get("question") or "",
                constraints.get("review_issue") or "",
                constraints.get("review_feedback") or "",
                " ".join(str(x) for x in (result.get("constraint_issues") or [])),
            ]
        )
        q = Counter(tokenize(query))
        current_ids = [str(x) for x in (pack.get("images") or [])]
        rows = list(self.records_by_manual.get(manual_id) or [])
        for pic in pack.get("pic_evidence") or []:
            if record_image_id(pic) and record_image_id(pic) not in self.records_by_id:
                rows.append(pic)
        scored = []
        for rec in rows:
            image_id = record_image_id(rec)
            text = record_text(rec)
            score = 0.0
            counts = Counter(tokenize(text))
            for tok, count in q.items():
                if tok in counts:
                    score += min(count, 3) * min(counts[tok], 3)
            if image_id in current_ids:
                score += 6
            if score > 0 or image_id in current_ids:
                scored.append((score, image_key(image_id), rec))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[dict[str, Any]] = []
        seen = set()
        for _score, _key, rec in scored:
            image_id = record_image_id(rec)
            if image_id in seen:
                continue
            seen.add(image_id)
            selected.append(rec)
            if len(selected) >= limit:
                break
        for image_id in current_ids:
            rec = self.records_by_id.get(image_id)
            if rec and image_id not in seen:
                selected.append(rec)
                seen.add(image_id)
        selected.sort(key=lambda rec: image_key(record_image_id(rec)))
        return selected

    def repair_one(self, rid: str, model: str, timeout: float) -> dict[str, Any]:
        started = time.perf_counter()
        pack = self.packs[rid]
        result = self.results[rid]
        constraints = pack.get("answer_constraints") or {}
        candidates = self.candidate_records(pack, result)
        candidate_lines = []
        for rec in candidates:
            candidate_lines.append(
                f"- {record_image_id(rec)} | caption={record_text(rec)[:500]}"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You repair image selection for product-manual Q&A. "
                    "Choose only image IDs from the candidate list. Return compact JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Select the final image IDs for this question.\n"
                    "Rules:\n"
                    "- Prefer no image over a weak or wrong image.\n"
                    "- Keep images only when a sentence in the answer can directly explain that image.\n"
                    "- Do not include neighboring chapter images, storage/maintenance images, or cross-manual images unless the question asks for them.\n"
                    "- Preserve the natural manual order.\n"
                    "- Return JSON: {\"selected_images\": [\"...\"], \"reason\": \"...\"}.\n\n"
                    f"Question:\n{pack.get('question')}\n\n"
                    f"Current selected images:\n{pack.get('images') or []}\n\n"
                    f"Review constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
                    f"Verifier failure issues:\n{json.dumps(result.get('constraint_issues') or [], ensure_ascii=False)}\n\n"
                    "Candidate images:\n"
                    + "\n".join(candidate_lines[:44])
                ),
            },
        ]
        raw = deepseek_chat(messages, model, timeout)
        data = extract_json_object(raw)
        selected = [str(x) for x in (data.get("selected_images") or []) if str(x).strip()]
        allowed = {record_image_id(rec) for rec in candidates}
        selected = list(dict.fromkeys(image_id for image_id in selected if image_id in allowed))
        return {
            "id": rid,
            "selected_images": selected,
            "reason": str(data.get("reason") or "").strip(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "ok": True,
        }

    def merged_overrides(self, repairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        repairs_by_id = {str(row["id"]): row for row in repairs if row.get("ok")}
        out = []
        seen = set()
        for row in self.original_override_rows:
            rid = str(row.get("id"))
            seen.add(rid)
            repair = repairs_by_id.get(rid)
            if repair:
                row = dict(row)
                row["raw_selected_images_before_constraint_repair"] = row.get("selected_images") or []
                row["selected_images"] = repair["selected_images"]
                row["source"] = "deepseek_constraint_failure_repair"
                row["reason"] = repair["reason"]
                notes = [str(x) for x in (row.get("postprocess_notes") or [])]
                notes.append("repaired_after_constraint_failure")
                row["postprocess_notes"] = list(dict.fromkeys(notes))
            out.append(row)
        for rid, repair in repairs_by_id.items():
            if rid not in seen:
                pack = self.packs[rid]
                constraints = pack.get("answer_constraints") or {}
                out.append(
                    {
                        "id": rid,
                        "manual_id": constraints.get("manual_id") or "",
                        "question": pack.get("question") or "",
                        "review_action": constraints.get("action") or "",
                        "review_risk": constraints.get("risk") or "",
                        "current_images": pack.get("images") or [],
                        "selected_images": repair["selected_images"],
                        "source": "deepseek_constraint_failure_repair",
                        "reason": repair["reason"],
                        "postprocess_notes": ["repaired_after_constraint_failure"],
                    }
                )
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--base-overrides", default=str(ASSET_DIR / "question_review_image_overrides_final.jsonl"))
    parser.add_argument("--output", default=str(ASSET_DIR / "question_review_image_overrides_final_constrained.jsonl"))
    parser.add_argument("--repair-log", default=str(ASSET_DIR / "question_review_image_repair_log.jsonl"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("DEEPSEEK_TIMEOUT", "45")))
    args = parser.parse_args()

    repairer = ImageRepairer(Path(args.packs), Path(args.results), Path(args.base_overrides))
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        ids = repairer.failing_ids(args.limit)
    repairs = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(repairer.repair_one, rid, args.model, args.timeout): rid for rid in ids}
        for future in as_completed(future_map):
            rid = future_map[future]
            try:
                repair = future.result()
            except Exception as exc:
                repair = {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            repairs.append(repair)
            print(f"[{'ok' if repair.get('ok') else 'fail'}] {rid} -> {repair.get('selected_images')} {repair.get('error','')}", flush=True)
    repairs.sort(key=lambda row: int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]))
    dump_jsonl(Path(args.repair_log), repairs)
    merged = repairer.merged_overrides(repairs)
    dump_jsonl(Path(args.output), merged)
    print(json.dumps({"repaired": sum(1 for row in repairs if row.get("ok")), "failed": sum(1 for row in repairs if not row.get("ok")), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
