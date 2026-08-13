from __future__ import annotations

import hashlib
import json
from typing import Any

from .context import current_request_context


def versioned_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    body = {
        "context": current_request_context().cache_namespace(),
        "payload": payload,
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"
