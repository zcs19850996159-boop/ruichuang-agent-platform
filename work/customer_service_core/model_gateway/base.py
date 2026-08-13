from __future__ import annotations

from typing import Any

from ..context import RequestContext


class UnsupportedModelOperation(NotImplementedError):
    pass


class BaseModelGateway:
    """Provider-neutral gateway contract.

    Legacy DeepSeek/Qwen calls are migrated behind concrete adapters
    incrementally so the competition behavior stays reproducible.
    """

    def generate(self, prompt: str, *, context: RequestContext, **options: Any) -> str:
        raise UnsupportedModelOperation("generate adapter is not configured")

    def stream_generate(
        self,
        prompt: str,
        *,
        context: RequestContext,
        **options: Any,
    ) -> Any:
        raise UnsupportedModelOperation("stream_generate adapter is not configured")

    def vision_analyze(
        self,
        images: list[dict[str, Any]],
        *,
        context: RequestContext,
        **options: Any,
    ) -> dict[str, Any]:
        raise UnsupportedModelOperation("vision_analyze adapter is not configured")

    def embed(self, texts: list[str], *, context: RequestContext, **options: Any) -> Any:
        raise UnsupportedModelOperation("embed adapter is not configured")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        context: RequestContext,
        **options: Any,
    ) -> list[dict[str, Any]]:
        raise UnsupportedModelOperation("rerank adapter is not configured")
