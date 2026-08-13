from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://u488863-aq5n-23e9383f.westd.seetacloud.com:8443/chat"
DEFAULT_INPUT = str(Path("work") / "a_rank_question_route_gold.csv")
DEFAULT_REFERENCE_RET = str(Path("work") / "canonical_highscore_reference_v62_base81625.csv")
DEFAULT_PREFIX = "a_rank_public_regression_afterfix"

IMAGE_ID_RE = re.compile(
    r"\b(?:Manual\d+|Camera|drill\d*|jetski|Security_Camera|air_conditioner|exercise_bikes|fitness_trackers|oven|fax|generator|Dish_washer|Blower|function_keyboard|toothbrush\d*|vr|VR)[A-Za-z0-9]*_[A-Za-z0-9]+\b"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            text = raw.decode(enc)
            rows = list(csv.DictReader(text.splitlines()))
            if rows:
                return rows
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode CSV: {path}")


def clean_question(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.replace('","', "\n").strip()


def parse_gold_images(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    text = text.strip().strip('"')
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip().strip('"').strip("'") for part in text.split(",") if part.strip().strip('"').strip("'")]


def parse_reference_ret_images(value: str) -> list[str]:
    text = str(value or "").strip().strip('"')
    if not text:
        return []
    match = re.search(r",\s*(\[[^\]]*\])\s*$", text, flags=re.S)
    if not match:
        return []
    raw = match.group(1)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [m.group(0) for m in IMAGE_ID_RE.finditer(raw)]


def load_reference_ret(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(row.get("id") or ""): str(row.get("ret") or "") for row in read_rows(path)}


def infer_manual_from_images(images: list[str]) -> str:
    counts: Counter[str] = Counter()
    for image_id in images:
        match = re.match(r"(Manual\d+)_", str(image_id or ""))
        if match:
            counts[match.group(1)] += 1
    if len(counts) == 1:
        return counts.most_common(1)[0][0]
    return ""


def is_english(text: str) -> bool:
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return letters >= 3 and letters > cjk


def f1(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def expected_route(row: dict[str, str], gold_images: list[str] | None = None) -> tuple[str, str]:
    gold_manual = str(row.get("gold_manual") or "").strip()
    route_type = str(row.get("route_type") or "").strip()
    if route_type == "policy_service" or gold_manual == "none_policy":
        return "policy_service", "none_policy"
    inferred_manual = infer_manual_from_images(gold_images or [])
    if inferred_manual:
        return "manual", inferred_manual
    return "manual", gold_manual


def call_chat(url: str, row: dict[str, str], timeout: int, retries: int, verify_ssl: bool, api_token: str = "") -> dict[str, Any]:
    row_id = str(row.get("id") or "")
    question = clean_question(row.get("question") or "")
    payload = {"question": question, "session_id": f"a-rank-regression-{row_id}", "use_memory": False}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    context = None if verify_ssl else ssl._create_unverified_context()
    started = time.time()
    last_error = ""
    for attempt in range(retries + 1):
        try:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                data["_http_status"] = getattr(resp, "status", 200)
                data["_wall_ms"] = round((time.time() - started) * 1000, 2)
                return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {"code": -1, "msg": last_error, "data": {}, "_http_status": 0, "_wall_ms": round((time.time() - started) * 1000, 2)}


def evaluate(row: dict[str, str], response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data") or {}
    route = data.get("route") or {}
    answer_check = data.get("answer_check") or {}
    answer = str(data.get("answer") or "")
    images = [str(x) for x in (data.get("images") or [])]
    gold_images = parse_reference_ret_images(row.get("reference_ret") or "")
    gold_manual = str(row.get("gold_manual") or "").strip()
    reference_manual = infer_manual_from_images(gold_images)
    if gold_manual.startswith("Manual") and reference_manual and reference_manual != gold_manual:
        teacher_images = parse_gold_images(row.get("teacher_image_ids") or "")
        if teacher_images:
            gold_images = teacher_images
    if not gold_images and not row.get("reference_ret"):
        gold_images = parse_gold_images(row.get("teacher_image_ids") or "")
    exp_route_type, exp_manual = expected_route(row, gold_images)
    route_type = str(route.get("route_type") or "")
    manual_id = str(route.get("manual_id") or "")
    pic_count = answer.count("<PIC>")
    image_count = len(images)
    if isinstance(answer_check.get("pic_count"), int):
        pic_count = int(answer_check["pic_count"])
    if isinstance(answer_check.get("image_count"), int):
        image_count = int(answer_check["image_count"])
    precision, recall, image_f1 = f1(images, gold_images)
    leaked = [m.group(0) for m in IMAGE_ID_RE.finditer(answer.replace("<PIC>", ""))]
    route_type_ok = route_type == exp_route_type
    manual_ok = manual_id == exp_manual if exp_manual else True
    image_exact_set = set(images) == set(gold_images)
    image_exact_order = images == gold_images
    pic_match_images = pic_count == image_count
    policy_has_images = exp_route_type == "policy_service" and image_count > 0
    english_contains_cjk = is_english(clean_question(row.get("question") or "")) and any("\u4e00" <= ch <= "\u9fff" for ch in answer)
    issues: list[str] = []
    if int(response.get("code", -1)) != 0:
        issues.append("api_error")
    if not route_type_ok:
        issues.append("route_type_mismatch")
    if not manual_ok:
        issues.append("manual_mismatch")
    if not pic_match_images:
        issues.append("pic_image_count_mismatch")
    if not bool(answer_check.get("constraint_pass", False)):
        issues.append("constraint_failed")
    if leaked:
        issues.append("image_id_leak_in_answer")
    if policy_has_images:
        issues.append("policy_answer_has_images")
    if english_contains_cjk:
        issues.append("english_answer_contains_cjk")
    if gold_images and not images:
        issues.append("gold_images_missing")
    if gold_images and image_f1 < 0.5:
        issues.append("low_image_f1")
    return {
        "id": str(row.get("id") or ""),
        "question": clean_question(row.get("question") or ""),
        "expected_route_type": exp_route_type,
        "expected_manual": exp_manual,
        "gold_images": gold_images,
        "product_or_policy": row.get("product_or_policy", ""),
        "intent_type": row.get("intent_type", ""),
        "code": response.get("code", -1),
        "msg": response.get("msg", ""),
        "http_status": response.get("_http_status", 0),
        "route_type": route_type,
        "manual_id": manual_id,
        "route_reason": route.get("reason", ""),
        "route_confidence": route.get("confidence", ""),
        "images": images,
        "answer": answer,
        "ret": data.get("ret", ""),
        "constraint_pass": bool(answer_check.get("constraint_pass", False)),
        "constraint_issues": answer_check.get("constraint_issues") or [],
        "pic_count": pic_count,
        "image_count": image_count,
        "pic_match_images": pic_match_images,
        "route_type_ok": route_type_ok,
        "manual_ok": manual_ok,
        "image_precision": round(precision, 4),
        "image_recall": round(recall, 4),
        "image_f1": round(image_f1, 4),
        "image_exact_set": image_exact_set,
        "image_exact_order": image_exact_order,
        "leaked_image_ids": leaked,
        "elapsed_ms": data.get("elapsed_ms", ""),
        "wall_ms": response.get("_wall_ms", ""),
        "issues": issues,
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    issue_counts: Counter[str] = Counter()
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    by_manual: dict[str, Counter[str]] = defaultdict(Counter)
    sums = Counter()
    image_f1_sum = 0.0
    image_precision_sum = 0.0
    image_recall_sum = 0.0
    image_rows = 0
    for item in items:
        exp_route = item["expected_route_type"]
        exp_manual = item["expected_manual"]
        by_route[exp_route]["total"] += 1
        by_manual[exp_manual]["total"] += 1
        for key in ("route_type_ok", "manual_ok", "constraint_pass", "pic_match_images", "image_exact_set", "image_exact_order"):
            if item.get(key):
                sums[key] += 1
                by_route[exp_route][key] += 1
                by_manual[exp_manual][key] += 1
        if item["code"] == 0:
            sums["api_ok"] += 1
        if item["gold_images"]:
            image_rows += 1
            image_f1_sum += float(item["image_f1"])
            image_precision_sum += float(item["image_precision"])
            image_recall_sum += float(item["image_recall"])
        for issue in item["issues"]:
            issue_counts[issue] += 1
            by_route[exp_route][f"issue:{issue}"] += 1
            by_manual[exp_manual][f"issue:{issue}"] += 1

    def rate(n: int, d: int = total) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "total": total,
        "api_ok": sums["api_ok"],
        "api_ok_rate": rate(sums["api_ok"]),
        "route_type_ok": sums["route_type_ok"],
        "route_type_accuracy": rate(sums["route_type_ok"]),
        "manual_ok": sums["manual_ok"],
        "manual_accuracy": rate(sums["manual_ok"]),
        "constraint_pass": sums["constraint_pass"],
        "constraint_pass_rate": rate(sums["constraint_pass"]),
        "pic_match_images": sums["pic_match_images"],
        "pic_match_images_rate": rate(sums["pic_match_images"]),
        "image_exact_set": sums["image_exact_set"],
        "image_exact_set_rate": rate(sums["image_exact_set"]),
        "image_exact_order": sums["image_exact_order"],
        "image_exact_order_rate": rate(sums["image_exact_order"]),
        "gold_image_rows": image_rows,
        "image_avg_precision": round(image_precision_sum / image_rows, 4) if image_rows else 0.0,
        "image_avg_recall": round(image_recall_sum / image_rows, 4) if image_rows else 0.0,
        "image_avg_f1": round(image_f1_sum / image_rows, 4) if image_rows else 0.0,
        "issue_counts": dict(issue_counts.most_common()),
        "by_expected_route": {key: dict(value) for key, value in sorted(by_route.items())},
        "by_expected_manual": {key: dict(value) for key, value in sorted(by_manual.items())},
    }


def write_outputs(output_dir: Path, prefix: str, items: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}_answers.jsonl"
    detail_path = output_dir / f"{prefix}_details.csv"
    issue_path = output_dir / f"{prefix}_issues.csv"
    submission_path = output_dir / f"{prefix}_submission.csv"
    summary_path = output_dir / f"{prefix}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    fields = [
        "id",
        "expected_route_type",
        "expected_manual",
        "route_type",
        "manual_id",
        "route_type_ok",
        "manual_ok",
        "constraint_pass",
        "pic_count",
        "image_count",
        "pic_match_images",
        "image_f1",
        "image_exact_set",
        "image_exact_order",
        "issues",
        "gold_images",
        "images",
        "constraint_issues",
        "answer",
        "question",
    ]
    for path, only_issues in ((detail_path, False), (issue_path, True)):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in items:
                if only_issues and not item["issues"]:
                    continue
                row = {key: item.get(key, "") for key in fields}
                for key in ("issues", "gold_images", "images", "constraint_issues"):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                writer.writerow(row)

    with submission_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "ret"])
        writer.writeheader()
        for item in items:
            writer.writerow({"id": item["id"], "question": item["question"], "ret": item.get("ret") or ""})

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {issue_path}")
    print(f"Wrote {submission_path}")
    print(f"Wrote {summary_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=str(Path("outputs") / "rag_agent"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--reference-ret", default=DEFAULT_REFERENCE_RET)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--api-token", default=os.environ.get("KAFU_API_TOKEN", ""))
    args = parser.parse_args(argv)

    rows = read_rows(Path(args.input))
    reference_ret = load_reference_ret(Path(args.reference_ret)) if args.reference_ret else {}
    if reference_ret:
        for row in rows:
            row["reference_ret"] = reference_ret.get(str(row.get("id") or ""), "")
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("id") or "") in wanted]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded rows: {len(rows)}")
    print(f"Workers: {args.workers}")
    print(f"URL: {args.url}")
    started = time.time()
    items_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(call_chat, args.url, row, args.timeout, args.retries, not args.insecure, args.api_token): row
            for row in rows
        }
        for idx, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                response = future.result()
            except Exception as exc:
                response = {"code": -1, "msg": repr(exc), "data": {}, "_http_status": 0, "_wall_ms": 0}
            item = evaluate(row, response)
            items_by_id[item["id"]] = item
            if idx == 1 or idx % 10 == 0 or idx == len(rows):
                elapsed = time.time() - started
                print(
                    f"Progress {idx}/{len(rows)} elapsed={elapsed:.1f}s "
                    f"id={item['id']} route={item['route_type']}/{item['manual_id']} "
                    f"img_f1={item['image_f1']} issues={','.join(item['issues'][:3])}"
                )
                sys.stdout.flush()

    items = [items_by_id[str(row.get("id") or "")] for row in rows if str(row.get("id") or "") in items_by_id]
    summary = summarize(items)
    write_outputs(Path(args.output_dir), args.prefix, items, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
