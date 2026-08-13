from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..interfaces import ModelGateway


class OpenAICompatibleModelGateway(ModelGateway):
    """Provider-neutral HTTP gateway for one-pass grounded text generation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.timeout = timeout or float(os.environ.get("DEEPSEEK_TIMEOUT", "75"))

    def generate(self, messages: list[dict[str, str]], **options: Any) -> str:
        if not self.api_key.strip():
            raise RuntimeError("model API key is not configured")
        payload = {
            "model": options.get("model") or self.model,
            "messages": messages,
            "temperature": float(options.get("temperature", 0.1)),
            "max_tokens": int(options.get("max_tokens", 1800)),
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"model gateway HTTP {exc.code}: {detail}") from exc
        return str(result["choices"][0]["message"]["content"]).strip()

    def stream_generate(self, messages: list[dict[str, str]], **options: Any):
        raise NotImplementedError("managed knowledge streaming is not enabled in Tool API v1")

    def vision_analyze(self, image: Any, **options: Any) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, texts: list[str], **options: Any) -> list[list[float]]:
        raise NotImplementedError

    def rerank(self, query: str, candidates: list[str], **options: Any) -> list[float]:
        raise NotImplementedError
