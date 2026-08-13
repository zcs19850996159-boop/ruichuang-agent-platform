from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

from visual_grounding import VerifiedVisualGrounding


def main() -> None:
    raw = b"verified-image-bytes"
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        rules_path = Path(tmp) / "rules.json"
        rules_path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "rule_id": "fixture",
                            "sha256": digest,
                            "sha256_aliases": [hashlib.sha256(b"alias-bytes").hexdigest()],
                            "source_urls": ["https://example.test/video"],
                            "review_status": "approved",
                            "manual_id": "Manual99",
                            "retrieval_query": "verified query",
                            "structured": {"product": "fixture product", "vision_confidence": 0.2},
                            "answer_zh": "verified answer",
                            "image_ids": ["Manual99_1"],
                        },
                        {
                            "rule_id": "not-approved",
                            "sha256": "0" * 64,
                            "review_status": "pending",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        os.environ["VERIFIED_VISUAL_GROUNDING_RULES"] = str(rules_path)
        matcher = VerifiedVisualGrounding()
        assert matcher.status()["rule_count"] == 1
        result = matcher.match(
            [{"index": 1, "base64": base64.b64encode(raw).decode("ascii")}]
        )
        assert result["matched"] is True
        assert result["manual_id"] == "Manual99"
        assert result["structured"]["vision_confidence"] == 1.0
        assert result["image_ids"] == ["Manual99_1"]
        alias = matcher.match(
            [{"index": 1, "base64": base64.b64encode(b"alias-bytes").decode("ascii")}]
        )
        assert alias["rule_id"] == "fixture"
        source_match = matcher.match(
            [
                {
                    "index": 1,
                    "base64": base64.b64encode(b"unknown-transcode").decode("ascii"),
                    "source_url": "https://example.test/video",
                }
            ]
        )
        assert source_match["rule_id"] == "fixture"
        assert matcher.match([{"index": 1, "base64": base64.b64encode(b"other").decode("ascii")}])[
            "matched"
        ] is False
    print("visual grounding tests passed")


if __name__ == "__main__":
    main()
