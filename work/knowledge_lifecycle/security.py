from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = int(os.environ.get("KNOWLEDGE_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"ignore\s+(all\s+)?(?:(previous|prior)\s+)?(?:system\s+)?(instructions?|prompts?)",
        re.I,
    ),
    re.compile(r"(system|developer)\s*prompt", re.I),
    re.compile(r"忽略.{0,12}(系统|之前|以上).{0,12}(指令|提示)", re.I),
    re.compile(r"(发送|上传|泄露).{0,20}(用户|密钥|密码|隐私).{0,20}(接口|网址|服务器)", re.I),
)


@dataclass(frozen=True, slots=True)
class FileInspection:
    path: Path
    extension: str
    mime: str
    size_bytes: int
    sha256: str
    virus_scan: str


def _detect_type(path: Path) -> str:
    head = path.read_bytes()[:8]
    suffix = path.suffix.lower()
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"PK\x03\x04"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix in {".txt", ".md"}:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("text documents must be valid UTF-8") from exc
        return "text/markdown" if suffix == ".md" else "text/plain"
    return "application/octet-stream"


def inspect_upload(path: str | Path) -> FileInspection:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError("source file does not exist")
    extension = candidate.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported manual type: {extension or '(none)'}")
    size = candidate.stat().st_size
    if size <= 0:
        raise ValueError("manual is empty")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"manual exceeds {MAX_UPLOAD_BYTES} byte limit")
    mime = _detect_type(candidate)
    expected = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }[extension]
    if mime != expected:
        raise ValueError(f"file content does not match extension {extension}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    scanner = shutil.which("clamscan")
    require_scan = os.environ.get("KNOWLEDGE_REQUIRE_VIRUS_SCAN", "0") == "1"
    virus_scan = "not_available"
    if scanner:
        completed = subprocess.run(
            [scanner, "--no-summary", str(candidate)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode == 1:
            raise ValueError("malware scanner rejected the document")
        if completed.returncode > 1:
            raise RuntimeError("malware scanner failed")
        virus_scan = "passed"
    elif require_scan:
        raise RuntimeError("virus scanning is required but no scanner is installed")
    return FileInspection(candidate, extension, mime, size, digest.hexdigest(), virus_scan)


def find_prompt_injection(text: str) -> list[str]:
    findings: list[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(match.group(0)[:160])
    return findings
