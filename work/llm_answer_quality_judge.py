from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWERS = ROOT / "outputs" / "rag_agent" / "full400_final_v73_20260620" / "full400_final_v73_20260620_answers.jsonl"
DEFAULT_REFERENCE = ROOT / "work" / "canonical_highscore_reference_v62_base81625.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rag_agent" / "llm_answer_quality_judge"

IMAGE_ID_RE = re.compile(
    r"\b(?:Manual\d+|Camera|drill\d*|jetski|Security_Camera|air_conditioner|exercise_bikes|fitness_trackers|oven|fax|generator|Dish_washer|Blower|function_keyboard|toothbrush\d*|vr|VR)[A-Za-z0-9]*_[A-Za-z0-9]+\b"
)

SCORE_FIELDS = [
    "coverage",
    "faithfulness",
    "image_correctness",
    "multi_intent",
    "language_style",
    "overall",
]


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compact(text: Any, limit: int = 2800) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def parse_image_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [m.group(0) for m in IMAGE_ID_RE.finditer(text)]


def split_ret(ret: str) -> tuple[str, list[str]]:
    text = str(ret or "").strip()
    if not text:
        return "", []
    match = re.search(r",\s*(\[[^\]]*\])\s*$", text, flags=re.S)
    if not match:
        return strip_outer_quote(text), parse_image_list(text)
    answer = strip_outer_quote(text[: match.start()].strip())
    images = parse_image_list(match.group(1))
    return answer, images


def strip_outer_quote(text: str) -> str:
    value = str(text or "").strip()
    for quote in ('"', "'", "“", "”"):
        if value.startswith(quote):
            value = value[1:].strip()
    for quote in ('"', "'", "“", "”"):
        if value.endswith(quote):
            value = value[:-1].strip()
    return value


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        row_id = str(row.get("id") or row.get("ID") or "").strip()
        if not row_id:
            continue
        answer, images = split_ret(row.get("ret") or "")
        out[row_id] = {
            "id": row_id,
            "reference_ret": row.get("ret") or "",
            "reference_answer": answer,
            "reference_images": images,
        }
    return out


def f1(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    precision = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    recall = tp / len(gs) if gs else (1.0 if not ps else 0.0)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def extract_system_images(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if isinstance(images, list):
        return [str(x).strip() for x in images if str(x).strip()]
    if row.get("ret"):
        return split_ret(str(row.get("ret") or ""))[1]
    return []


def build_prompt(row: dict[str, Any], ref: dict[str, Any]) -> list[dict[str, str]]:
    system_images = extract_system_images(row)
    ref_images = [str(x) for x in ref.get("reference_images") or []]
    precision, recall, image_f1 = f1(system_images, ref_images)
    route_type = str(row.get("route_type") or row.get("expected_route_type") or "")
    question = str(row.get("question") or "")
    system_answer = str(row.get("answer") or "")
    reference_answer = str(ref.get("reference_answer") or "")
    precheck = {
        "route_type": route_type,
        "system_manual_id": row.get("manual_id") or "",
        "system_image_ids": system_images,
        "reference_image_ids": ref_images,
        "pic_count": row.get("pic_count", system_answer.count("<PIC>")),
        "system_image_count": len(system_images),
        "reference_image_count": len(ref_images),
        "image_precision": round(precision, 4),
        "image_recall": round(recall, 4),
        "image_f1": round(image_f1, 4),
        "constraint_pass": row.get("constraint_pass", ""),
    }
    user = f"""
你是多模态客服智能体比赛的离线质量评审员。请只评估“系统回答质量”，不要改写答案，不要输出额外解释。

正式参考答案只使用下面的 0.81625 高分参考答案。系统答案不要求逐字相同，但必须在事实、关键点、图片配图和语言风格上达到等价或接近等价。

评分维度均为 0-5 分：
- coverage：是否覆盖用户问题和参考答案的关键点。
- faithfulness：是否忠实于参考答案/手册/客服规则，有无幻觉、跑题或错误手册内容。
- image_correctness：系统图片 ID、<PIC> 位置和数量是否与参考答案及正文内容匹配。
- multi_intent：如果问题包含多个意图/多个子问题，是否逐一回答；单意图题可给 5。
- language_style：语言是否符合题目语言、客服/手册场景和正式提交风格。
- overall：综合质量。

判定规则：
- pass=true 表示该答案作为提交答案基本可接受。
- 如果存在漏答、错答、严重漏图、英文题中文答、客服题话术明显不像官方样例、或“当前证据不足”误拒答，pass=false。
- 若参考答案图片 ID 与题面明显疑似冲突，可在 reference_conflict_possible=true 标注；但仍需以本参考答案作为评分锚点。
- 输出必须是合法 JSON，不要 Markdown，不要代码块。

待评审样本：
ID: {row.get("id")}
题目: {compact(question, 1200)}
自动预检: {json.dumps(precheck, ensure_ascii=False)}

系统回答:
{compact(system_answer, 3200)}

0.81625参考答案:
{compact(reference_answer, 3600)}

请输出 JSON，字段必须包含：
{{
  "coverage": 0-5,
  "faithfulness": 0-5,
  "image_correctness": 0-5,
  "multi_intent": 0-5,
  "language_style": 0-5,
  "overall": 0-5,
  "hallucination_risk": "low|medium|high",
  "major_issue": "一句话说明主要问题；无重大问题写none",
  "pass": true|false,
  "reference_conflict_possible": true|false
}}
""".strip()
    return [
        {
            "role": "system",
            "content": "You are a strict answer-quality judge. Return valid JSON only. Do not rewrite or repair the answer.",
        },
        {"role": "user", "content": user},
    ]


def call_deepseek(messages: list[dict[str, str]], *, timeout: float, retries: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("JUDGE_MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("JUDGE_TEMPERATURE", "0.0")),
        "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "900")),
        "response_format": {"type": "json_object"},
        "thinking": {"type": os.environ.get("DEEPSEEK_THINKING", "disabled")},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"DeepSeek judge call failed: {last_error!r}")


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge output is not a JSON object")
    for field in SCORE_FIELDS:
        value = data.get(field, 0)
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            score = 0
        data[field] = max(0, min(5, score))
    risk = str(data.get("hallucination_risk") or "medium").lower()
    data["hallucination_risk"] = risk if risk in {"low", "medium", "high"} else "medium"
    data["major_issue"] = str(data.get("major_issue") or "none").strip()[:500]
    data["pass"] = bool(data.get("pass"))
    data["reference_conflict_possible"] = bool(data.get("reference_conflict_possible", False))
    return data


def judge_one(row: dict[str, Any], ref: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "id": str(row.get("id") or ""),
        "question": row.get("question") or "",
        "route_type": row.get("route_type") or row.get("expected_route_type") or "",
        "manual_id": row.get("manual_id") or "",
        "system_images": extract_system_images(row),
        "reference_images": [str(x) for x in ref.get("reference_images") or []],
        "system_answer": row.get("answer") or "",
        "reference_answer": ref.get("reference_answer") or "",
    }
    raw = call_deepseek(build_prompt(row, ref), timeout=timeout, retries=retries)
    parsed = parse_judge_json(raw)
    result.update(parsed)
    result["raw_judge"] = raw
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    score_sums = Counter()
    risk_counts = Counter()
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for field in SCORE_FIELDS:
            score_sums[field] += int(row.get(field) or 0)
        risk_counts[str(row.get("hallucination_risk") or "unknown")] += 1
        route = str(row.get("route_type") or "unknown")
        by_route[route]["total"] += 1
        if row.get("pass"):
            by_route[route]["pass"] += 1
        for field in SCORE_FIELDS:
            by_route[route][f"avg_{field}_sum"] += int(row.get(field) or 0)
    summary = {
        "total": total,
        "pass": sum(1 for row in rows if row.get("pass")),
        "pass_rate": round(sum(1 for row in rows if row.get("pass")) / total, 4) if total else 0.0,
        "reference_conflict_possible": sum(1 for row in rows if row.get("reference_conflict_possible")),
        "hallucination_risk_counts": dict(risk_counts),
        "average_scores": {
            field: round(score_sums[field] / total, 4) if total else 0.0 for field in SCORE_FIELDS
        },
        "low_score_counts": {
            field: sum(1 for row in rows if int(row.get(field) or 0) < 4) for field in SCORE_FIELDS
        },
        "by_route": {},
    }
    for route, counter in sorted(by_route.items()):
        route_total = counter["total"]
        summary["by_route"][route] = {
            "total": route_total,
            "pass": counter["pass"],
            "pass_rate": round(counter["pass"] / route_total, 4) if route_total else 0.0,
            "average_scores": {
                field: round(counter[f"avg_{field}_sum"] / route_total, 4) if route_total else 0.0
                for field in SCORE_FIELDS
            },
        }
    return summary


def write_outputs(output_dir: Path, prefix: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{prefix}_judge.jsonl"
    csv_path = output_dir / f"{prefix}_judge.csv"
    issue_path = output_dir / f"{prefix}_judge_issues.csv"
    summary_path = output_dir / f"{prefix}_judge_summary.json"
    md_path = output_dir / f"{prefix}_judge_report.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = [
        "id",
        "route_type",
        "manual_id",
        *SCORE_FIELDS,
        "hallucination_risk",
        "pass",
        "reference_conflict_possible",
        "major_issue",
        "system_images",
        "reference_images",
        "question",
    ]
    for path, issues_only in ((csv_path, False), (issue_path, True)):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if issues_only and row.get("pass") and not row.get("reference_conflict_possible"):
                    continue
                item = {field: row.get(field, "") for field in fields}
                item["system_images"] = json.dumps(item["system_images"], ensure_ascii=False)
                item["reference_images"] = json.dumps(item["reference_images"], ensure_ascii=False)
                writer.writerow(item)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, rows), encoding="utf-8")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {issue_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


def render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    worst = sorted(
        rows,
        key=lambda r: (int(r.get("overall") or 0), int(r.get("coverage") or 0), int(r.get("image_correctness") or 0)),
    )[:12]
    lines = [
        "# LLM-as-a-Judge 离线质量验证报告",
        "",
        "本报告用于提交前质量审查，不参与线上 `/chat` 回答生成，也不用于生成后修补答案。正式答案锚点仅使用 0.81625 高分参考答案。",
        "",
        "## 总体指标",
        "",
        f"- 样本数：{summary.get('total', 0)}",
        f"- Judge 通过：{summary.get('pass', 0)} / {summary.get('total', 0)}，通过率 {summary.get('pass_rate', 0):.2%}",
        f"- 疑似参考答案冲突：{summary.get('reference_conflict_possible', 0)}",
        f"- 幻觉风险分布：{json.dumps(summary.get('hallucination_risk_counts', {}), ensure_ascii=False)}",
        "",
        "## 平均得分",
        "",
    ]
    for field, value in (summary.get("average_scores") or {}).items():
        lines.append(f"- {field}: {value:.2f} / 5")
    lines.extend(["", "## 低分项统计", ""])
    for field, value in (summary.get("low_score_counts") or {}).items():
        lines.append(f"- {field} < 4：{value}")
    lines.extend(["", "## 主要问题样例", ""])
    if not worst:
        lines.append("无。")
    for row in worst:
        lines.append(
            f"- ID {row.get('id')}：overall={row.get('overall')}，pass={row.get('pass')}，"
            f"issue={row.get('major_issue')}，question={compact(row.get('question'), 120)}"
        )
    lines.append("")
    return "\n".join(lines)


def select_rows(
    answers: list[dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    ids: str,
    offset: int,
    limit: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected = answers
    if ids:
        wanted = {item.strip() for item in ids.split(",") if item.strip()}
        selected = [row for row in selected if str(row.get("id") or "") in wanted]
    if offset:
        selected = selected[offset:]
    if limit:
        selected = selected[:limit]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in selected:
        row_id = str(row.get("id") or "")
        ref = reference.get(row_id)
        if ref:
            pairs.append((row, ref))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline LLM-as-a-Judge quality audit for generated answers.")
    parser.add_argument("--answers", default=str(DEFAULT_ANSWERS), help="System answers JSONL from regression test.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="The only official high-score reference CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="full400_v73_sample")
    parser.add_argument("--ids", default="", help="Comma-separated IDs to judge.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args(argv)

    answers = load_jsonl(Path(args.answers))
    reference = load_reference(Path(args.reference))
    pairs = select_rows(answers, reference, ids=args.ids, offset=args.offset, limit=args.limit)
    if not pairs:
        raise RuntimeError("No rows selected. Check --answers, --reference, --ids, --offset, and --limit.")
    print(f"Loaded answers={len(answers)} reference={len(reference)} selected={len(pairs)}")
    print("Judge mode: offline evaluation only; no answer repair will be performed.")
    started = time.time()
    rows_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(judge_one, row, ref, args.timeout, args.retries): str(row.get("id") or "")
            for row, ref in pairs
        }
        for idx, future in enumerate(as_completed(futures), 1):
            row_id = futures[future]
            try:
                judged = future.result()
            except Exception as exc:
                judged = {
                    "id": row_id,
                    "question": "",
                    "route_type": "",
                    "manual_id": "",
                    "system_images": [],
                    "reference_images": [],
                    "system_answer": "",
                    "reference_answer": "",
                    "coverage": 0,
                    "faithfulness": 0,
                    "image_correctness": 0,
                    "multi_intent": 0,
                    "language_style": 0,
                    "overall": 0,
                    "hallucination_risk": "high",
                    "major_issue": f"judge_error: {exc!r}",
                    "pass": False,
                    "reference_conflict_possible": False,
                    "raw_judge": "",
                    "elapsed_ms": 0,
                }
            rows_by_id[row_id] = judged
            print(
                f"Progress {idx}/{len(pairs)} id={row_id} overall={judged.get('overall')} "
                f"pass={judged.get('pass')} issue={compact(judged.get('major_issue'), 80)}",
                flush=True,
            )
    order = [str(row.get("id") or "") for row, _ in pairs]
    rows = [rows_by_id[row_id] for row_id in order if row_id in rows_by_id]
    summary = summarize(rows)
    write_outputs(Path(args.output_dir), args.prefix, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
