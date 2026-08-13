from __future__ import annotations

from typing import Any, Callable, Protocol

from .context import RequestContext


class QueryRouter(Protocol):
    def route(self, request: dict[str, Any], context: RequestContext) -> dict[str, Any]: ...


class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, context: RequestContext) -> dict[str, Any]: ...


class MultimodalAnalyzer(Protocol):
    def analyze(self, attachments: list[dict[str, Any]], context: RequestContext) -> dict[str, Any]: ...


class AnswerGenerator(Protocol):
    def generate(
        self,
        evidence: dict[str, Any],
        context: RequestContext,
        stream_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...


class AnswerValidator(Protocol):
    def validate(
        self,
        answer: dict[str, Any],
        evidence: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]: ...


class ModelGateway(Protocol):
    def generate(self, prompt: str, *, context: RequestContext, **options: Any) -> str: ...

    def stream_generate(
        self,
        prompt: str,
        *,
        context: RequestContext,
        **options: Any,
    ) -> Any: ...

    def vision_analyze(
        self,
        images: list[dict[str, Any]],
        *,
        context: RequestContext,
        **options: Any,
    ) -> dict[str, Any]: ...

    def embed(self, texts: list[str], *, context: RequestContext, **options: Any) -> Any: ...

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        context: RequestContext,
        **options: Any,
    ) -> list[dict[str, Any]]: ...
