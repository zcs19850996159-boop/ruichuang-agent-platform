from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class ToolResult:
    tool: str
    data: dict[str, Any]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = ""
    elapsed_ms: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": self.tool,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "elapsed_ms": self.elapsed_ms,
            "data": self.data,
        }


def require_schema(arguments: dict[str, Any]) -> None:
    version = str(arguments.get("schema_version") or SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")


def timed_result(
    tool: str,
    data: dict[str, Any],
    *,
    started: float,
    request_id: str,
    trace_id: str,
) -> dict[str, Any]:
    return ToolResult(
        tool=tool,
        data=data,
        request_id=request_id,
        trace_id=trace_id,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    ).as_dict()
