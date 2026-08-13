from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass(slots=True)
class ExtractedDocument:
    title: str
    pages: list[str]
    images: list[tuple[str, bytes, int | None]] = field(default_factory=list)
    image_references: list[str] = field(default_factory=list)
    tables: int = 0
    warnings: list[str] = field(default_factory=list)
    parser: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)


def _extract_docx(path: Path) -> ExtractedDocument:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        try:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        except (KeyError, ElementTree.ParseError) as exc:
            raise ValueError("invalid DOCX document") from exc
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
            if text:
                paragraphs.append(text)
        images = [
            (Path(name).name, archive.read(name), None)
            for name in archive.namelist()
            if name.startswith("word/media/") and not name.endswith("/")
        ]
        tables = len(root.findall(".//w:tbl", namespace))
    return ExtractedDocument(
        title=path.stem,
        pages=["\n".join(paragraphs)],
        images=images,
        tables=tables,
        parser="docx-stdlib",
    )


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None
    if fitz is not None:
        document = fitz.open(path)
        pages: list[str] = []
        images: list[tuple[str, bytes, int | None]] = []
        for page_number, page in enumerate(document, start=1):
            pages.append(page.get_text("text").strip())
            for image_number, image in enumerate(page.get_images(full=True), start=1):
                payload = document.extract_image(image[0])
                extension = str(payload.get("ext") or "bin")
                images.append(
                    (
                        f"page-{page_number}-image-{image_number}.{extension}",
                        payload["image"],
                        page_number,
                    )
                )
        warnings = []
        blank_pages = sum(not page.strip() for page in pages)
        if pages and blank_pages / len(pages) >= 0.3:
            provider = os.environ.get("KNOWLEDGE_OCR_PROVIDER", "").strip().lower()
            if provider:
                if provider != "tesseract":
                    document.close()
                    raise RuntimeError(f"unsupported OCR provider: {provider}")
                executable = shutil.which("tesseract")
                if not executable:
                    document.close()
                    raise RuntimeError("tesseract OCR provider is configured but unavailable")
                language = os.environ.get("KNOWLEDGE_OCR_LANGUAGE", "chi_sim+eng")
                for index, text in enumerate(pages):
                    if text.strip():
                        continue
                    pixmap = document[index].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    completed = subprocess.run(
                        [executable, "stdin", "stdout", "-l", language],
                        input=pixmap.tobytes("png"),
                        capture_output=True,
                        timeout=120,
                        check=False,
                    )
                    if completed.returncode == 0:
                        pages[index] = completed.stdout.decode("utf-8", errors="replace").strip()
                warnings.append("ocr_applied")
                if sum(not page.strip() for page in pages) / len(pages) >= 0.3:
                    warnings.append("scanned_or_low_text_pdf")
            else:
                warnings.append("scanned_or_low_text_pdf")
        document.close()
        return ExtractedDocument(path.stem, pages, images, warnings=warnings, parser="pymupdf")
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF or pypdf") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    warnings = ["pdf_images_not_extracted"]
    if pages and sum(not page for page in pages) / len(pages) >= 0.3:
        warnings.append("scanned_or_low_text_pdf")
    return ExtractedDocument(path.stem, pages, warnings=warnings, parser="pypdf")


def extract_document(path: str | Path) -> ExtractedDocument:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = source.read_text(encoding="utf-8")
        image_references: list[str] = []
        parser = "utf8-text"
        warnings: list[str] = []
        if suffix == ".txt":
            try:
                envelope = json.loads(text)
            except json.JSONDecodeError:
                envelope = None
            if (
                isinstance(envelope, list)
                and len(envelope) == 2
                and isinstance(envelope[0], str)
                and isinstance(envelope[1], list)
                and all(isinstance(item, str) and item.strip() for item in envelope[1])
            ):
                text = envelope[0]
                image_references = [item.strip() for item in envelope[1]]
                parser = "official-json-text-v1"
                if text.count("<PIC>") != len(image_references):
                    warnings.append("official_image_marker_count_mismatch")
        pages = [part.strip() for part in re.split(r"\f|\n\s*---page---\s*\n", text) if part.strip()]
        return ExtractedDocument(
            source.stem,
            pages or [text],
            image_references=image_references,
            warnings=warnings,
            parser=parser,
        )
    if suffix == ".docx":
        return _extract_docx(source)
    if suffix == ".pdf":
        return _extract_pdf(source)
    raise ValueError(f"unsupported document type: {suffix}")
