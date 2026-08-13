from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_INPUT = r"C:\Users\admin\Downloads\synthetic_200_new_knowledge_questions_audit.csv"
DEFAULT_PREFIX = "synthetic_200_new_audit_public"

EN_SOURCE_MANUAL_MAP = {
    "Coffee machine manual": "Manual07",
    "Air fryer manual": "Manual08",
    "Pressure cooker air fryer manual": "Manual30",
    "Boat manual": "Manual09",
    "Camera manual": "Manual10",
    "Earphones manual": "Manual12",
    "eReader manual": "Manual13",
    "Fax manual": "Manual15",
    "Grill manual": "Manual19",
    "Jetski manual": "Manual20",
    "Landline manual": "Manual22",
    "Lawn mower manual": "Manual23",
    "Microwave manual": "Manual24",
    "Motherboard manual": "Manual25",
    "Pressure cooker manual": "Manual30",
    "Robot vacuum manual": "Manual32",
    "Snowmobile manual": "Manual34",
    "TV/radio manual": "Manual35",
}

CN_MANUAL_SUFFIX = "\u624b\u518c"
SERVICE_POLICY = "service_policy"
CN_MANUAL = "cn_manual"
EN_MANUAL = "en_manual"

REFUSAL_PATTERNS = [
    "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3",
    "\u8bc1\u636e\u4e0d\u8db3",
    "\u65e0\u6cd5\u6839\u636e\u624b\u518c",
    "\u672a\u63d0\u4f9b\u5177\u4f53\u5185\u5bb9",
    "does not contain",
    "provided manual evidence does not contain",
    "insufficient evidence",
]

IMAGE_ID_RE = re.compile(
    r"\b(?:Manual\d+|Camera|drill\d*|jetski|Security_Camera|air_conditioner|exercise_bikes|fitness_trackers|oven|fax|generator|Dish_washer|Blower|function_keyboard|vr|VR|Manual)[A-Za-z0-9]*_[A-Za-z0-9]+\b"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def load_manifest_manual_map(root: Path) -> dict[str, str]:
    candidates = [
        root / "outputs" / "rag_assets" / "manifest.json",
        root / "work" / "customer_agent_submission" / "rag_assets" / "manifest.json",
    ]
    manual_map: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for manual_id, info in data.get("manuals", {}).items():
            product = str(info.get("product") or "").strip()
            source_file = str(info.get("source_file") or "").strip()
            if product:
                manual_map[product + CN_MANUAL_SUFFIX] = manual_id
                manual_map[product] = manual_id
            if source_file.endswith(".txt"):
                manual_map[source_file[:-4]] = manual_id
        break
    manual_map.update(EN_SOURCE_MANUAL_MAP)
    return manual_map


def expected_for(row: dict[str, str], manual_map: dict[str, str]) -> tuple[str, str]:
    category = (row.get("category") or "").strip()
    source = (row.get("source_manual") or "").strip()
    if category == SERVICE_POLICY:
        return "policy_service", "none_policy"
    if category in (CN_MANUAL, EN_MANUAL):
        return "manual", manual_map.get(source, "")
    return "", manual_map.get(source, "")


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def extract_topic_terms(row: dict[str, str]) -> list[str]:
    raw = " ".join(
        str(row.get(key) or "")
        for key in ("knowledge_point", "source_manual")
    )
    terms: list[str] = []
    for term in re.split(r"[/,;|()\[\]\s]+", raw):
        term = term.strip(" -_:")
        if len(term) >= 2 and term not in {"manual", "OK"}:
            terms.append(term)
    return terms


def call_chat(url: str, row: dict[str, str], timeout: int, retries: int, verify_ssl: bool) -> dict[str, Any]:
    row_id = str(row.get("id") or "")
    question = str(row.get("question") or "")
    payload = {
        "question": question,
        "session_id": f"generalization-{row_id}",
        "use_memory": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    context = None if verify_ssl else ssl._create_unverified_context()
    last_error = ""
    started = time.time()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                body = resp.read().decode("utf-8")
                parsed = json.loads(body)
                parsed["_http_status"] = getattr(resp, "status", 200)
                parsed["_wall_ms"] = round((time.time() - started) * 1000, 2)
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
    return {
        "code": -1,
        "msg": last_error,
        "data": {},
        "_http_status": 0,
        "_wall_ms": round((time.time() - started) * 1000, 2),
    }


def evaluate_one(
    row: dict[str, str],
    response: dict[str, Any],
    manual_map: dict[str, str],
) -> dict[str, Any]:
    data = response.get("data") or {}
    route = data.get("route") or {}
    answer = str(data.get("answer") or "")
    images = data.get("images") or []
    answer_check = data.get("answer_check") or {}
    expected_route, expected_manual = expected_for(row, manual_map)
    route_type = str(route.get("route_type") or "")
    manual_id = str(route.get("manual_id") or "")
    pic_count = answer.count("<PIC>")
    image_count = len(images) if isinstance(images, list) else 0
    check_pic_count = answer_check.get("pic_count")
    check_image_count = answer_check.get("image_count")
    if isinstance(check_pic_count, int):
        pic_count = check_pic_count
    if isinstance(check_image_count, int):
        image_count = check_image_count

    category = (row.get("category") or "").strip()
    leaked_ids = [m.group(0) for m in IMAGE_ID_RE.finditer(answer.replace("<PIC>", ""))]
    refused = any(pat.lower() in answer.lower() for pat in REFUSAL_PATTERNS)
    terms = extract_topic_terms(row)
    hit_terms = [term for term in terms if term and term.lower() in answer.lower()]
    route_type_correct = (not expected_route) or route_type == expected_route
    manual_correct = True
    if expected_manual:
        manual_correct = manual_id == expected_manual
    elif category in (CN_MANUAL, EN_MANUAL):
        manual_correct = False

    issues: list[str] = []
    if int(response.get("code", -1)) != 0:
        issues.append("api_error")
    if not route_type_correct:
        issues.append("route_type_mismatch")
    if not manual_correct:
        issues.append("manual_mismatch")
    if not bool(answer_check.get("constraint_pass", False)):
        issues.append("constraint_failed")
    if pic_count != image_count:
        issues.append("pic_image_count_mismatch")
    if leaked_ids:
        issues.append("image_id_leak_in_answer")
    if category == EN_MANUAL and has_cjk(answer):
        issues.append("english_answer_contains_cjk")
    if category == SERVICE_POLICY and image_count:
        issues.append("policy_answer_has_images")
    if category in (CN_MANUAL, EN_MANUAL) and image_count == 0:
        issues.append("manual_answer_no_images_warning")
    if refused:
        issues.append("refusal_or_insufficient_evidence")
    if category == SERVICE_POLICY and not hit_terms:
        issues.append("low_specificity_warning")

    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "category": category,
        "source_manual": row.get("source_manual", ""),
        "knowledge_point": row.get("knowledge_point", ""),
        "max_similarity_to_existing": row.get("max_similarity_to_existing", ""),
        "expected_route_type": expected_route,
        "expected_manual": expected_manual,
        "http_status": response.get("_http_status", 0),
        "code": response.get("code", -1),
        "msg": response.get("msg", ""),
        "route_type": route_type,
        "manual_id": manual_id,
        "route_confidence": route.get("confidence", ""),
        "route_reason": route.get("reason", ""),
        "route_type_correct": route_type_correct,
        "manual_correct": manual_correct,
        "constraint_pass": bool(answer_check.get("constraint_pass", False)),
        "constraint_issues": answer_check.get("constraint_issues", []),
        "pic_count": pic_count,
        "image_count": image_count,
        "pic_match_images": pic_count == image_count,
        "images": images,
        "ret": data.get("ret", ""),
        "answer": answer,
        "elapsed_ms": data.get("elapsed_ms", ""),
        "wall_ms": response.get("_wall_ms", ""),
        "refused": refused,
        "leaked_image_ids": leaked_ids,
        "topic_terms": terms,
        "topic_term_hits": hit_terms,
        "issues": issues,
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(details)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_manual: dict[str, Counter[str]] = defaultdict(Counter)
    issue_counts: Counter[str] = Counter()
    route_ok = manual_ok = constraints = pic_ok = code_ok = 0
    manual_rows = 0
    manual_with_images = 0
    en_rows = 0
    en_lang_ok = 0
    service_rows = 0
    service_specific = 0
    for item in details:
        cat = item["category"]
        src = item["source_manual"]
        by_category[cat]["total"] += 1
        by_manual[src]["total"] += 1
        if item["code"] == 0:
            code_ok += 1
            by_category[cat]["api_ok"] += 1
        if item["route_type_correct"]:
            route_ok += 1
            by_category[cat]["route_type_correct"] += 1
        if item["manual_correct"]:
            manual_ok += 1
            by_category[cat]["manual_correct"] += 1
            by_manual[src]["manual_correct"] += 1
        if item["constraint_pass"]:
            constraints += 1
            by_category[cat]["constraint_pass"] += 1
        if item["pic_match_images"]:
            pic_ok += 1
            by_category[cat]["pic_match_images"] += 1
        if cat in (CN_MANUAL, EN_MANUAL):
            manual_rows += 1
            if item["image_count"] > 0:
                manual_with_images += 1
                by_category[cat]["with_images"] += 1
        if cat == EN_MANUAL:
            en_rows += 1
            if "english_answer_contains_cjk" not in item["issues"]:
                en_lang_ok += 1
                by_category[cat]["language_ok"] += 1
        if cat == SERVICE_POLICY:
            service_rows += 1
            if "low_specificity_warning" not in item["issues"]:
                service_specific += 1
                by_category[cat]["specificity_ok"] += 1
        for issue in item["issues"]:
            issue_counts[issue] += 1
            by_category[cat][f"issue:{issue}"] += 1
            by_manual[src][f"issue:{issue}"] += 1

    def rate(n: int, d: int = total) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "total": total,
        "api_ok": code_ok,
        "api_ok_rate": rate(code_ok),
        "route_type_correct": route_ok,
        "route_type_accuracy": rate(route_ok),
        "manual_correct": manual_ok,
        "manual_accuracy": rate(manual_ok),
        "constraint_pass": constraints,
        "constraint_pass_rate": rate(constraints),
        "pic_match_images": pic_ok,
        "pic_match_images_rate": rate(pic_ok),
        "manual_rows": manual_rows,
        "manual_with_images": manual_with_images,
        "manual_with_images_rate": rate(manual_with_images, manual_rows),
        "english_rows": en_rows,
        "english_language_ok_rate": rate(en_lang_ok, en_rows),
        "service_rows": service_rows,
        "service_specificity_ok_rate": rate(service_specific, service_rows),
        "issue_counts": dict(issue_counts.most_common()),
        "by_category": {
            key: dict(value)
            for key, value in sorted(by_category.items())
        },
        "by_source_manual": {
            key: dict(value)
            for key, value in sorted(by_manual.items())
        },
    }


def write_outputs(output_dir: Path, prefix: str, details: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}_answers.jsonl"
    details_csv_path = output_dir / f"{prefix}_details.csv"
    issues_csv_path = output_dir / f"{prefix}_issues.csv"
    submission_path = output_dir / f"{prefix}_submission.csv"
    summary_path = output_dir / f"{prefix}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    detail_fields = [
        "id",
        "category",
        "source_manual",
        "knowledge_point",
        "expected_route_type",
        "expected_manual",
        "code",
        "route_type",
        "manual_id",
        "route_type_correct",
        "manual_correct",
        "constraint_pass",
        "pic_count",
        "image_count",
        "pic_match_images",
        "refused",
        "issues",
        "images",
        "answer",
        "ret",
        "question",
    ]
    with details_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for item in details:
            row = {key: item.get(key, "") for key in detail_fields}
            row["issues"] = json.dumps(row["issues"], ensure_ascii=False)
            row["images"] = json.dumps(row["images"], ensure_ascii=False)
            writer.writerow(row)

    with issues_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for item in details:
            if item["issues"]:
                row = {key: item.get(key, "") for key in detail_fields}
                row["issues"] = json.dumps(row["issues"], ensure_ascii=False)
                row["images"] = json.dumps(row["images"], ensure_ascii=False)
                writer.writerow(row)

    with submission_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "ret"])
        writer.writeheader()
        for item in details:
            writer.writerow(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "ret": item.get("ret") or json.dumps([item["answer"], item["images"]], ensure_ascii=False),
                }
            )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {details_csv_path}")
    print(f"Wrote {issues_csv_path}")
    print(f"Wrote {submission_path}")
    print(f"Wrote {summary_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=str(Path("outputs") / "rag_agent"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args(argv)

    root = Path.cwd()
    input_path = Path(args.input)
    rows = read_csv_rows(input_path)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    manual_map = load_manifest_manual_map(root)
    print(f"Loaded rows: {len(rows)}")
    print(f"Workers: {args.workers}")
    print(f"URL: {args.url}")

    details_by_id: dict[str, dict[str, Any]] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(call_chat, args.url, row, args.timeout, args.retries, not args.insecure): row
            for row in rows
        }
        for idx, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                response = future.result()
            except Exception as exc:
                response = {"code": -1, "msg": repr(exc), "data": {}, "_http_status": 0, "_wall_ms": 0}
            item = evaluate_one(row, response, manual_map)
            details_by_id[str(item["id"])] = item
            if idx == 1 or idx % 10 == 0 or idx == len(rows):
                elapsed = time.time() - started
                print(
                    f"Progress {idx}/{len(rows)} elapsed={elapsed:.1f}s "
                    f"id={item['id']} code={item['code']} route={item['route_type']}/{item['manual_id']} "
                    f"issues={','.join(item['issues'][:3])}"
                )
                sys.stdout.flush()

    original_order = [str(row.get("id") or "") for row in rows]
    details = [details_by_id[row_id] for row_id in original_order if row_id in details_by_id]
    summary = summarize(details)
    write_outputs(Path(args.output_dir), args.prefix, details, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
