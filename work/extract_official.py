from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "work" / "official_1165.html"


def extract_js_string(source: str, key: str) -> str:
    marker = f"{key}:"
    start = source.index(marker) + len(marker)
    while source[start].isspace():
        start += 1
    if source[start] != '"':
        raise ValueError(f"{key} is not a double-quoted JS string")
    pos = start + 1
    escaped = False
    chunks: list[str] = []
    while pos < len(source):
        ch = source[pos]
        if escaped:
            chunks.append("\\" + ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            break
        else:
            chunks.append(ch)
        pos += 1
    return json.loads('"' + "".join(chunks) + '"')


def normalize_text(markdown: str) -> str:
    markdown = html.unescape(markdown)
    markdown = re.sub(r"<br\s*/?>", "\n", markdown, flags=re.I)
    markdown = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", markdown, flags=re.I)
    markdown = re.sub(r"</tr>\s*<tr[^>]*>", "\n", markdown, flags=re.I)
    markdown = re.sub(r"<[^>]+>", "", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def main() -> None:
    source = HTML_PATH.read_text(encoding="utf-8")
    description = normalize_text(extract_js_string(source, "cmptDescription"))
    data_description = normalize_text(extract_js_string(source, "cmptDataDescription"))
    agreement = normalize_text(extract_js_string(source, "cmptAgreement"))

    (ROOT / "work" / "official_1165_description.md").write_text(description, encoding="utf-8")
    (ROOT / "work" / "official_1165_data_description.md").write_text(data_description, encoding="utf-8")
    (ROOT / "work" / "official_1165_agreement.md").write_text(agreement, encoding="utf-8")

    print("HEADINGS")
    for line in (description + "\n" + data_description).splitlines():
        if line.startswith("#"):
            print(line)

    print("\nKEY_LINES")
    keywords = re.compile(
        r"评分|评审|验证|报告|代码|文档|提交|答案|接口|/chat|question|images|Bearer|超时|Base64|多模态|幻觉|知识库"
    )
    for line in (description + "\n" + data_description).splitlines():
        clean = line.strip()
        if clean and keywords.search(clean):
            print(clean[:300])


if __name__ == "__main__":
    main()
