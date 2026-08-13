from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


ERROR_CODE_RE = re.compile(
    r"\b(?:E\d{1,4}|F\d{1,4}|ERR[-_ ]?[A-Z0-9]+|[A-Z]{1,4}[-_]\d{1,4})\b",
    re.I,
)
BUTTON_RE = re.compile(
    r"\b(?:power|menu|reset|start|stop|mode|set|ok|on/off)\b|"
    r"(?:电源|菜单|复位|启动|停止|模式|设置|确认)键?",
    re.I,
)


def _manual_context(image: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> str:
    texts = [
        str(chunks_by_id[chunk_id].get("text") or "").strip()
        for chunk_id in image.get("bound_chunk_ids") or []
        if chunk_id in chunks_by_id
    ]
    return " ".join(texts)[:800]


def _ocr(path: Path) -> tuple[str, float, str]:
    provider = os.environ.get("KNOWLEDGE_IMAGE_OCR_PROVIDER", "").strip().lower()
    if not provider:
        return "", 0.0, "not_configured"
    if provider != "tesseract":
        return "", 0.0, "unsupported_provider"
    executable = shutil.which("tesseract")
    if not executable:
        return "", 0.0, "provider_unavailable"
    language = os.environ.get("KNOWLEDGE_OCR_LANGUAGE", "chi_sim+eng")
    completed = subprocess.run(
        [executable, str(path), "stdout", "-l", language],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return "", 0.0, "ocr_failed"
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    return text, (0.75 if text else 0.0), "tesseract"


def enrich_stage_images(
    stage: Path,
    chunks: list[dict[str, Any]],
    images: list[dict[str, Any]],
    *,
    product_id: str = "",
) -> dict[str, int]:
    """Enrich new images during offline staging; never blocks on an unconfigured VLM."""
    chunks_by_id = {
        str(chunk.get("chunk_id") or ""): chunk
        for chunk in chunks
    }
    stats = {"total": 0, "enriched": 0, "ocr_ready": 0, "requires_review": 0}
    for image in images:
        if product_id and str(image.get("product_id") or "") != product_id:
            continue
        relative = str(image.get("file") or "")
        if not relative:
            continue
        path = stage / relative
        if not path.is_file():
            continue
        stats["total"] += 1
        if isinstance(image.get("structured"), dict):
            stats["enriched"] += 1
            continue
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        context = _manual_context(image, chunks_by_id)
        ocr_text, ocr_confidence, ocr_provider = _ocr(path)
        vision_result: dict[str, Any] = {}
        vision_error = ""
        if os.environ.get("KNOWLEDGE_IMAGE_VISION_ENABLED", "0") == "1":
            try:
                from build_image_knowledge import call_vision

                vision_result = call_vision(
                    raw,
                    mimetypes.guess_type(path.name)[0] or "image/jpeg",
                    {
                        **image,
                        "section_title": context[:300],
                    },
                    os.environ.get("VISION_BASE_URL", "http://127.0.0.1:8001/v1"),
                    os.environ.get(
                        "VISION_MODEL",
                        "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct",
                    ),
                    os.environ.get("VISION_API_KEY", "local-key"),
                )
            except Exception as exc:
                vision_error = f"{type(exc).__name__}: {exc}"[:300]
        if vision_result.get("ocr_text"):
            ocr_text = str(vision_result["ocr_text"])
            ocr_confidence = float(vision_result.get("ocr_confidence") or 0.0)
            ocr_provider = "vision_model"
        searchable = f"{ocr_text} {context}".strip()
        structured = {
            "ocr_text": ocr_text,
            "caption": str(vision_result.get("caption") or context[:500]),
            "visual_tags": sorted(
                set(str(item) for item in vision_result.get("visual_tags") or [])
                | set(
                    re.findall(
                        r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,}",
                        searchable,
                    )
                )
            )[:24],
            "components": list(vision_result.get("components") or []),
            "buttons": sorted(
                set(str(item) for item in vision_result.get("buttons") or [])
                | set(BUTTON_RE.findall(searchable))
            )[:16],
            "error_codes": sorted(
                set(str(item) for item in vision_result.get("error_codes") or [])
                | set(ERROR_CODE_RE.findall(searchable))
            )[:16],
            "screen_state": str(vision_result.get("screen_state") or ""),
            "vision_confidence": float(
                vision_result.get("vision_confidence")
                or (0.35 if context else 0.0)
            ),
            "ocr_confidence": ocr_confidence,
            "ocr_provider": ocr_provider,
            "caption_source": (
                "vision_model"
                if vision_result
                else "bound_manual_context"
                if context
                else "none"
            ),
            "vision_error": vision_error,
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
        image.update(
            {
                "source_hash": source_hash,
                "chunk_version": f"image-auto-v2-{source_hash[:12]}",
                "review_status": (
                    "machine_generated"
                    if vision_result or ocr_text
                    else "requires_visual_review"
                ),
                "structured": structured,
            }
        )
        stats["enriched"] += 1
        stats["ocr_ready"] += int(bool(ocr_text))
        stats["requires_review"] += int(not bool(ocr_text))
    return stats
