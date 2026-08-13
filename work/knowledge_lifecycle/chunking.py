from __future__ import annotations

import hashlib
import re
from typing import Any

from .extractors import ExtractedDocument


HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+|第[一二三四五六七八九十百\d]+[章节部分]|"
    r"\d+(?:\.\d+)+\s+\S|\d+[、.)）:：]\s*|[A-Z][A-Z0-9 /_-]{2,}$)"
)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 /_-]{2,}$")
STEP_RE = re.compile(
    r"^(?:步骤\s*[一二三四五六七八九十\d]+|step\s*\d+|"
    r"\d+[\s]*[.)、）]|[（(][一二三四五六七八九十\d]+[)）])",
    re.I,
)
ERROR_CODE_RE = re.compile(
    r"(?:错误码|故障码|error\s*code|fault\s*code|代码)\s*[:：#-]?\s*[A-Z0-9_-]{2,}",
    re.I,
)
WARNING_RE = re.compile(r"^(?:警告|注意|危险|提示|warning|caution|danger|note)\b", re.I)
TABLE_BORDER_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*[-+:| ]{5,}\s*$")


def _language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk and latin:
        return "zh-en"
    if cjk:
        return "zh"
    if latin:
        return "en"
    return "und"


def _classify_unit(text: str, *, allow_uppercase_heading: bool = True) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    if ERROR_CODE_RE.search(text):
        return "error_code"
    if WARNING_RE.search(first):
        return "warning"
    if STEP_RE.search(first) or sum(bool(STEP_RE.search(line)) for line in lines) >= 2:
        return "step"
    if any(TABLE_BORDER_RE.match(line) for line in lines) or sum("\t" in line for line in lines) >= 2:
        return "table"
    if (
        HEADING_RE.match(first)
        and len(text) <= 160
        and (allow_uppercase_heading or not UPPER_HEADING_RE.fullmatch(first))
    ):
        return "heading"
    return "paragraph"


def _structural_units(
    text: str,
    *,
    flush_on_blank: bool = True,
    allow_uppercase_headings: bool = True,
) -> list[tuple[str, str, str]]:
    """Split at semantic boundaries before applying the size budget.

    PDF and DOCX extractors commonly insert a blank line after every positioned
    text cell. Treating those layout blanks as paragraph boundaries produces
    dozens of tiny chunks and separates a step label from its explanation.
    Callers can therefore ignore blank lines for layout-derived text while the
    explicit heading, step, warning and error-code boundaries remain intact.
    """
    units: list[tuple[str, str, str]] = []
    section = ""
    buffer: list[str] = []
    buffer_type = "paragraph"

    def flush() -> None:
        nonlocal buffer
        value = "\n".join(buffer).strip()
        if value:
            units.append((buffer_type, value, section))
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if flush_on_blank:
                flush()
            continue
        line_type = _classify_unit(
            line,
            allow_uppercase_heading=allow_uppercase_headings,
        )
        if line_type == "heading":
            flush()
            section = line.lstrip("#").strip()
            units.append(("heading", line, section))
            buffer_type = "paragraph"
            continue
        if line_type in {"step", "error_code", "warning", "table"}:
            if buffer and buffer_type != line_type:
                flush()
            buffer_type = line_type
            buffer.append(line)
            if line_type in {"step", "error_code", "warning"}:
                flush()
                buffer_type = "paragraph"
            continue
        if buffer and buffer_type != "paragraph":
            flush()
        buffer_type = "paragraph"
        buffer.append(line)
    flush()
    return units


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    units = [
        part.strip()
        for part in re.split(r"\n{2,}|(?<=[。！？.!?])\s+|(?<=<PIC>)", text)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, max_chars - overlap)
            chunks.extend(unit[offset : offset + max_chars] for offset in range(0, len(unit), step))
            continue
        candidate = f"{current}\n{unit}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{unit}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_chunks(
    document: ExtractedDocument,
    *,
    document_id: str,
    product_id: str,
    max_chars: int = 1200,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    image_reference_index = 0
    effective_overlap = 0 if document.image_references else overlap
    layout_derived = document.parser in {"pymupdf", "pypdf", "docx-stdlib"}
    for page_number, page_text in enumerate(document.pages, start=1):
        heading = ""
        structured: list[tuple[str, str, str]] = []
        for unit_type, unit_text, unit_section in _structural_units(
            page_text,
            flush_on_blank=not layout_derived,
            allow_uppercase_headings=not layout_derived,
        ):
            parts = _split_text(unit_text, max_chars, effective_overlap)
            structured.extend((unit_type, part, unit_section) for part in parts)
        for chunk_type, text, unit_section in structured:
            first_line = text.splitlines()[0].strip()
            if unit_section:
                heading = unit_section
            elif len(first_line) <= 100 and HEADING_RE.match(first_line):
                heading = first_line
            core = {
                "document_id": document_id,
                "product_id": product_id,
                "page": page_number,
                "section_path": heading,
                "chunk_type": chunk_type,
                "language": _language(text),
                "text": text,
            }
            source_hash = hashlib.sha256(repr(sorted(core.items())).encode("utf-8")).hexdigest()
            chunk = {
                "chunk_id": source_hash[:20],
                "document_id": document_id,
                "product_id": product_id,
                "title": document.title,
                "page": page_number,
                "section": heading,
                "section_path": heading,
                "source_page": page_number,
                "chunk_type": chunk_type,
                "language": core["language"],
                "review_status": "source_extracted",
                "source_hash": source_hash,
                "chunk_version": f"structured-v3-{source_hash[:12]}",
                "text": text,
                "source_ref": f"{document.title}:page:{page_number}",
            }
            marker_count = text.count("<PIC>")
            if marker_count and document.image_references:
                chunk["image_ids"] = document.image_references[
                    image_reference_index : image_reference_index + marker_count
                ]
                image_reference_index += marker_count
            chunks.append(chunk)
    return chunks
