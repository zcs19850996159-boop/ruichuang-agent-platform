from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from build_own_evidence import EvidenceBuilder
from conversation_memory import ConversationMemoryManager
from dynamic_image_selector import DynamicImageSelector
from generate_own_answers import format_ret, run_one


ROOT = Path(__file__).resolve().parents[1]


def load_memory(path: Path) -> ConversationMemoryManager:
    manager = ConversationMemoryManager()
    if not path.exists():
        return manager
    data = json.loads(path.read_text(encoding="utf-8"))
    for session_id, raw in data.items():
        session = manager.get(session_id)
        session.active_route_type = raw.get("active_route_type") or ""
        session.active_manual_id = raw.get("active_manual_id") or ""
        session.active_product = raw.get("active_product") or ""
        session.active_policy_topics = [str(x) for x in raw.get("active_policy_topics") or []]
        session.last_user_question = raw.get("last_user_question") or ""
        session.last_resolved_question = raw.get("last_resolved_question") or ""
    return manager


def write_meta_cache(path: Path, selector_result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selector_result, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--question", required=True)
    parser.add_argument("--memory-store", default="outputs/rag_agent/agent_memory_store.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("DEEPSEEK_TIMEOUT", "75")))
    parser.add_argument("--dry-run", action="store_true", help="Resolve memory and select images, but do not call the answer model.")
    args = parser.parse_args()

    memory_path = ROOT / args.memory_store
    manager = load_memory(memory_path)
    resolved = manager.resolve_user_question(args.session_id, args.question)
    row_id = str(int(time.time() * 1000))
    question_for_rag = resolved["resolved_question"]
    manual_hint = resolved.get("manual_id_hint") or ""

    selector = DynamicImageSelector(use_llm=not args.dry_run, use_known_routes=False, candidate_k=50, timeout=args.timeout)
    selected = selector.select(row_id, question_for_rag, manual_hint=manual_hint)
    selected["id"] = row_id

    response: dict[str, Any] = {
        "memory": resolved,
        "selector": selected,
    }
    if not args.dry_run:
        cache_path = ROOT / "outputs/rag_agent/runtime_last_turn_selector.jsonl"
        write_meta_cache(cache_path, selected)
        old_cache = os.environ.get("META_IMAGE_SELECTION_CACHE")
        os.environ["META_IMAGE_SELECTION_CACHE"] = str(cache_path.relative_to(ROOT))
        try:
            builder = EvidenceBuilder()
            pack = builder.build(row_id, question_for_rag)
        finally:
            if old_cache is None:
                os.environ.pop("META_IMAGE_SELECTION_CACHE", None)
            else:
                os.environ["META_IMAGE_SELECTION_CACHE"] = old_cache
        result = run_one(pack, args.model, args.timeout)
        manager.add_assistant_answer(args.session_id, result.get("answer") or "")
        response["answer_result"] = result
        response["ret"] = format_ret(result.get("answer") or "", result.get("images") or [])

    manager.save(memory_path)
    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
