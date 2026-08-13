from __future__ import annotations

import json
import os
from typing import Any


class RedisStateStore:
    """Small JSON state adapter. Callers can keep their file fallback when Redis is unavailable."""

    def __init__(self, url: str = "", prefix: str = "customer-agent") -> None:
        self.url = url or os.environ.get("REDIS_URL", "")
        self.prefix = (prefix or "customer-agent").strip(":")
        self.client: Any = None
        self.error = ""
        self._connect()

    def _connect(self) -> bool:
        if not self.url:
            return False
        try:
            import redis

            self.client = redis.Redis.from_url(
                self.url,
                decode_responses=True,
                socket_connect_timeout=1.5,
                socket_timeout=2.0,
                health_check_interval=30,
            )
            self.client.ping()
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.client = None
            return False

    @property
    def ready(self) -> bool:
        if self.client is None and not self._connect():
            return False
        try:
            self.client.ping()
            self.error = ""
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.client = None
            return False

    def status(self) -> dict[str, Any]:
        ready = self.ready
        return {
            "backend": "redis" if ready else "file_fallback",
            "configured": bool(self.url),
            "ready": ready,
            "error": self.error,
        }

    def _key(self, namespace: str, key: str) -> str:
        return f"{self.prefix}:{namespace}:{key}"

    def get_json(self, namespace: str, key: str) -> Any:
        if not self.ready:
            return None
        try:
            raw = self.client.get(self._key(namespace, key))
            return json.loads(raw) if raw else None
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.client = None
            return None

    def set_json(self, namespace: str, key: str, value: Any, ttl_seconds: int = 0) -> None:
        if not self.ready:
            return
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        target = self._key(namespace, key)
        try:
            if ttl_seconds > 0:
                self.client.setex(target, ttl_seconds, raw)
            else:
                self.client.set(target, raw)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.client = None

    def delete(self, namespace: str, key: str) -> None:
        if self.ready:
            try:
                self.client.delete(self._key(namespace, key))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.client = None
