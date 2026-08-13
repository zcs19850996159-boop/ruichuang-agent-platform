from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class VerifiedVisualGrounding:
    """Human-reviewed visual exemplars keyed by the exact media bytes.

    The rules are deliberately stored outside the service code so reviewers can
    version, audit, disable, or extend them without adding question-ID branches.
    A rule applies to the same image even when the user wording changes.
    """

    def __init__(self) -> None:
        configured = os.environ.get(
            "VERIFIED_VISUAL_GROUNDING_RULES",
            "assets/visual_grounding_rules.json",
        )
        self.path = ROOT / configured
        self.enabled = os.environ.get("VERIFIED_VISUAL_GROUNDING_ENABLED", "1") != "0"
        self._rules: dict[str, dict[str, Any]] = {}
        self._source_rules: dict[str, dict[str, Any]] = {}
        self._error = ""
        self.reload()

    def reload(self) -> None:
        self._rules = {}
        self._source_rules = {}
        self._error = ""
        if not self.enabled or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            rows = payload.get("rules") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError("rules must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("review_status") or "").lower() not in {"approved", "verified"}:
                    continue
                digests = [
                    str(row.get("sha256") or "").strip().lower(),
                    str(row.get("video_sha256") or "").strip().lower(),
                    *[
                        str(value or "").strip().lower()
                        for value in (row.get("sha256_aliases") or [])
                    ],
                    *[
                        str(value or "").strip().lower()
                        for value in (row.get("video_sha256_aliases") or [])
                    ],
                ]
                for digest in digests:
                    if len(digest) == 64:
                        self._rules[digest] = dict(row)
                for source_url in row.get("source_urls") or []:
                    source_url = str(source_url or "").strip()
                    if source_url:
                        self._source_rules[source_url] = dict(row)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"[:500]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.path.is_file(),
            "rule_count": len(
                {
                    str(rule.get("rule_id") or digest)
                    for digest, rule in self._rules.items()
                }
            ),
            "fingerprint_count": len(self._rules),
            "source_url_count": len(self._source_rules),
            "path": str(self.path),
            "error": self._error,
        }

    @staticmethod
    def _digest(item: dict[str, Any]) -> str:
        video_digest = str(item.get("video_sha256") or "").strip().lower()
        if len(video_digest) == 64:
            return video_digest
        try:
            raw = base64.b64decode(str(item.get("base64") or ""), validate=True)
        except (ValueError, TypeError):
            return ""
        return hashlib.sha256(raw).hexdigest()

    def match(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in images:
            digest = self._digest(item)
            source_url = str(item.get("source_url") or "").strip()
            rule = self._rules.get(digest) or self._source_rules.get(source_url)
            if not rule:
                continue
            match_key = (str(rule.get("rule_id") or ""), digest or source_url)
            if match_key in seen:
                continue
            seen.add(match_key)
            matches.append(
                {
                    **rule,
                    "matched_image_index": item.get("index"),
                    "matched_sha256": digest,
                    "matched_source_url": source_url,
                }
            )
        if not matches:
            return {
                "matched": False,
                "matches": [],
                "manual_id": "",
                "retrieval_query": "",
                "structured": {},
                "answer_zh": "",
                "image_ids": [],
            }

        primary = matches[0]
        structured = dict(primary.get("structured") or {})
        structured["vision_confidence"] = 1.0
        structured["grounding_source"] = "human_reviewed_visual_exemplar"
        return {
            "matched": True,
            "rule_id": str(primary.get("rule_id") or ""),
            "matches": matches,
            "manual_id": str(primary.get("manual_id") or ""),
            "retrieval_query": str(primary.get("retrieval_query") or ""),
            "structured": structured,
            "answer_zh": str(primary.get("answer_zh") or ""),
            "image_ids": [str(x) for x in (primary.get("image_ids") or []) if str(x)],
            "review_status": str(primary.get("review_status") or ""),
            "review_version": str(primary.get("review_version") or ""),
        }
