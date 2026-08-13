from __future__ import annotations

from typing import Any, Callable


class LegacyAgentAdapter:
    """Temporary boundary around the frozen competition implementation.

    New REST, MCP, or WorkBuddy adapters call the application service. The
    application service reaches legacy code only through this adapter, allowing
    routing, retrieval, generation, and validation to be migrated independently
    under regression gates.
    """

    def __init__(self, implementation: Any) -> None:
        self._implementation = implementation

    def answer(
        self,
        payload: dict[str, Any],
        stream_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        return self._implementation.answer(payload, stream_callback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._implementation, name)


def create_legacy_runtime() -> LegacyAgentAdapter:
    # Delayed import prevents the compatibility implementation from becoming a
    # dependency of the core contracts at module-import time.
    from agent_api import AgentService

    return LegacyAgentAdapter(AgentService())
