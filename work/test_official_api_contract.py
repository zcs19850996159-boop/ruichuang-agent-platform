from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))


TOKEN = "contract-test-token"
TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def configure_env() -> None:
    os.environ["KAFU_API_TOKEN"] = TOKEN
    os.environ["API_AUTH_REQUIRED"] = "1"
    os.environ["API_ANSWER_CACHE"] = "0"
    os.environ["API_USE_LLM_SELECTOR"] = "0"
    os.environ["VISION_ENABLED"] = "1"
    os.environ["VISION_MOCK_DESCRIPTION"] = "a tiny red test image"
    os.environ["VISION_DIRECT_MIN_CONFIDENCE"] = "0"
    os.environ["AGENT_MEMORY_STORE"] = "outputs/rag_agent/test_contract_memory.json"


def request_json(
    base_url: str,
    payload: dict[str, Any],
    *,
    token: str | None = TOKEN,
    content_type: str = "application/json; charset=utf-8",
    request_id: str = "kf_req_contract_001",
    client_type: str = "web",
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": content_type}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if request_id:
        headers["X-Request-Id"] = request_id
    if client_type:
        headers["X-Client-Type"] = client_type
    req = urllib.request.Request(
        base_url + "/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main() -> None:
    configure_env()
    from agent_api import make_server

    server = make_server("127.0.0.1", 0)
    port = int(server.server_address[1])
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)

        status, body = request_json(base_url, {"question": "hello"}, token=None)
        assert status == 401, (status, body)
        assert body["code"] == 401

        status, body = request_json(base_url, {"question": "hello"}, token="wrong")
        assert status == 401, (status, body)

        status, body = request_json(base_url, {"question": "hello"})
        assert status == 200, (status, body)
        assert body["code"] == 0
        assert body["msg"] == "success"
        data = body["data"]
        assert data["answer"]
        assert data["session_id"].startswith("kf_session_")
        assert isinstance(data["timestamp"], int)
        assert data["request_id"] == "kf_req_contract_001"
        assert data["client_type"] == "web"

        status, body = request_json(
            base_url,
            {"question": "hello", "session_id": "kf_session_fixed"},
            request_id="kf_req_fixed",
            client_type="wx_miniprogram",
        )
        assert status == 200, (status, body)
        assert body["data"]["session_id"] == "kf_session_fixed"
        assert body["data"]["request_id"] == "kf_req_fixed"
        assert body["data"]["client_type"] == "wx_miniprogram"

        status, body = request_json(
            base_url,
            {"question": "Please describe this image.", "images": [TINY_PNG]},
        )
        assert status == 200, (status, body)
        assert body["data"]["route"]["route_type"] == "image_understanding"
        assert body["data"]["input_images"]["received"] == 1
        assert body["data"]["input_images"]["used"] is True

        status, body = request_json(
            base_url,
            {"question": "Please describe these images.", "images": [TINY_PNG, TINY_PNG, TINY_PNG, TINY_PNG]},
        )
        assert status == 400, (status, body)
        assert "at most 3" in body["msg"]

        status, body = request_json(
            base_url,
            {"question": "Please describe this image.", "images": [TINY_PNG.split(",", 1)[1]]},
        )
        assert status == 400, (status, body)
        assert "data:image" in body["msg"]

        status, body = request_json(
            base_url,
            {"question": "hello"},
            content_type="text/plain",
        )
        assert status == 415, (status, body)

        print("official API contract smoke test passed")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
