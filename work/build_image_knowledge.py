from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import mimetypes
import os
import socket
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "rag_assets" / "image_knowledge_auto.jsonl"
MAX_BYTES = 8 * 1024 * 1024


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
        return bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)
    except (OSError, ValueError):
        return False


def read_image(row: dict[str, Any]) -> tuple[bytes, str, str]:
    path_text = str(row.get("path") or "").strip()
    url = str(row.get("url") or "").strip()
    if path_text:
        path = Path(path_text).resolve()
        roots = [Path(x).resolve() for x in os.environ.get("IMAGE_ASSET_ROOTS", "/root/customer_agent_deploy").split(":") if x]
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError("local image path is outside IMAGE_ASSET_ROOTS")
        raw = path.read_bytes()
        source = str(path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    elif url:
        if not is_public_url(url):
            raise ValueError("image URL is not a public HTTP(S) address")
        req = request.Request(url, headers={"User-Agent": "customer-agent-image-builder/1.0"})
        with request.urlopen(req, timeout=20) as response:
            raw = response.read(MAX_BYTES + 1)
            mime = response.headers.get_content_type()
        source = url
    else:
        raise ValueError("manifest row requires path or url")
    if not raw or len(raw) > MAX_BYTES or not mime.startswith("image/"):
        raise ValueError("invalid or oversized image")
    return raw, mime, source


def parse_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    if start >= 0 and end < start:
        try:
            value = json.loads(cleaned[start:] + "}")
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    # A formatting failure must not block the whole offline build. Do not reinterpret malformed JSON as OCR.
    # Plain text may be kept as a low-confidence caption, while human-reviewed nearby_text remains a separate field.
    caption = cleaned[:500] if cleaned and not cleaned.startswith("{") else ""
    return {
        "ocr_text": "",
        "caption": caption,
        "visual_tags": [],
        "components": [],
        "buttons": [],
        "error_codes": [],
        "screen_state": "",
        "vision_confidence": 0.2 if caption else 0.0,
        "ocr_confidence": 0.0,
        "fallback_reason": "non_json_vision_response",
    }


def call_vision(raw: bytes, mime: str, row: dict[str, Any], base_url: str, model: str, api_key: str) -> dict[str, Any]:
    prompt = (
        "Analyze this product-manual image for an offline customer-service knowledge base. "
        "Return exactly one JSON object and no Markdown. Extract only visible facts. "
        "ocr_text must contain only text visibly present inside the image; never copy or infer wording from metadata. "
        "If no text is legible in the image, return an empty string for ocr_text. "
        'Schema: {"ocr_text":"","caption":"","visual_tags":[],"components":[],"buttons":[],'
        '"error_codes":[],"screen_state":"","vision_confidence":0.0,"ocr_confidence":0.0}. '
        "Preserve error-code spelling and button labels. Caption must explain the image in its manual context. "
        f"Known metadata: product={row.get('product', '')}; manual_id={row.get('manual_id', '')}; "
        f"section={row.get('section_title', '')}. "
        "Metadata is contextual and is not OCR evidence."
    )
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "temperature": 0,
        "max_tokens": 500,
    }
    req = request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    return parse_object(str(body["choices"][0]["message"]["content"]))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable OCR/caption/tag extraction for manual images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base-url", default=os.environ.get("VISION_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--model", default=os.environ.get("VISION_MODEL", "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--api-key", default=os.environ.get("VISION_API_KEY", "local-key"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_jsonl(Path(args.manifest))
    output = Path(args.output)
    completed = {
        str(row.get("source_hash") or "")
        for row in load_jsonl(output)
        if row.get("status") == "ok" and row.get("source_hash")
    }
    stats = {"manifest": len(manifest), "reused": 0, "processed": 0, "failed": 0}
    for row in manifest:
        if args.limit and stats["processed"] + stats["failed"] >= args.limit:
            break
        try:
            raw, mime, source = read_image(row)
            source_hash = hashlib.sha256(raw).hexdigest()
            if source_hash in completed:
                stats["reused"] += 1
                continue
            if args.dry_run:
                stats["processed"] += 1
                continue
            structured = call_vision(raw, mime, row, args.base_url, args.model, args.api_key)
            result = dict(row)
            result.update({
                "status": "ok",
                "source": source,
                "source_hash": source_hash,
                "chunk_version": "image-auto-v1-" + source_hash[:12],
                "review_status": "machine_generated",
                "structured": structured,
                "created_at": int(time.time()),
            })
            append_row(output, result)
            completed.add(source_hash)
            stats["processed"] += 1
        except Exception as exc:
            stats["failed"] += 1
            if not args.dry_run:
                append_row(output, {
                    **row,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": int(time.time()),
                })
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
