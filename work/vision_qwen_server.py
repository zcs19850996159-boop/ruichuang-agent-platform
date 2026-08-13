from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class QwenVisionEngine:
    def __init__(self, model_path: str, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.loaded_at = 0.0
        self.lock = threading.Lock()
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.process_vision_info: Any = None

    def load(self) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        min_pixels = parse_int_env("VISION_LOCAL_MIN_PIXELS", 256 * 28 * 28)
        max_pixels = parse_int_env("VISION_LOCAL_MAX_PIXELS", 768 * 28 * 28)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=os.environ.get("VISION_LOCAL_ATTN", "sdpa"),
        )
        self.model.eval()
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.loaded_at = time.time()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def status(self) -> dict[str, Any]:
        cuda = False
        gpu = ""
        memory_total_mb = None
        memory_free_mb = None
        if self.torch is not None:
            cuda = bool(self.torch.cuda.is_available())
            if cuda:
                gpu = self.torch.cuda.get_device_name(0)
                free, total = self.torch.cuda.mem_get_info(0)
                memory_total_mb = total // 1024**2
                memory_free_mb = free // 1024**2
        return {
            "loaded": self.loaded,
            "model": self.model_name,
            "model_path": self.model_path,
            "loaded_at": self.loaded_at,
            "cuda": cuda,
            "gpu": gpu,
            "memory_total_mb": memory_total_mb,
            "memory_free_mb": memory_free_mb,
        }

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("vision model is not loaded")
        messages = normalize_messages(payload.get("messages") or [])
        if not messages:
            raise ValueError("messages must contain at least one message")

        max_tokens = int(payload.get("max_tokens") or parse_int_env("VISION_LOCAL_MAX_NEW_TOKENS", 256))
        max_tokens = max(16, min(max_tokens, parse_int_env("VISION_LOCAL_MAX_NEW_TOKENS_HARD", 512)))
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        started = time.perf_counter()
        with self.lock:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        answer = re.sub(r"\s+", " ", answer)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs.input_ids.shape[-1]),
                "completion_tokens": int(generated_ids_trimmed[0].shape[-1]),
                "total_tokens": int(inputs.input_ids.shape[-1] + generated_ids_trimmed[0].shape[-1]),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def normalize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = raw.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content.strip():
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append({"type": "text", "text": text})
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url") or "").strip()
                    else:
                        url = str(image_url or "").strip()
                    if url:
                        parts.append({"type": "image", "image": url})
                elif item_type == "image":
                    image = str(item.get("image") or "").strip()
                    if image:
                        parts.append({"type": "image", "image": image})
        if parts:
            normalized.append({"role": role, "content": parts})
    return normalized


class VisionHandler(BaseHTTPRequestHandler):
    server_version = "QwenVisionCompat/1.0"

    @property
    def engine(self) -> QwenVisionEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("VISION_ACCESS_LOG", "0") == "1":
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(200, {"code": 0, "msg": "ok", "data": self.engine.status()})
            return
        if path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_name,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("empty request body")
            max_body = parse_int_env("VISION_LOCAL_MAX_REQUEST_BYTES", 16 * 1024 * 1024)
            if length > max_body:
                self.send_json(413, {"error": {"message": "request body too large"}})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("json body must be an object")
            result = self.engine.complete(payload)
        except Exception as exc:
            self.send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return
        self.send_json(200, result)


class VisionServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], engine: QwenVisionEngine) -> None:
        super().__init__(address, handler)
        self.engine = engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("VISION_LOCAL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VISION_LOCAL_PORT", "8001")))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VISION_LOCAL_MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--model-name", default=os.environ.get("VISION_LOCAL_MODEL_NAME", "Qwen2.5-VL-3B-Instruct"))
    args = parser.parse_args()

    engine = QwenVisionEngine(args.model_path, args.model_name)
    print(f"Loading vision model from {args.model_path}", flush=True)
    engine.load()
    print(json.dumps({"event": "vision_model_loaded", **engine.status()}, ensure_ascii=False), flush=True)
    server = VisionServer((args.host, args.port), VisionHandler, engine)
    print(f"Qwen vision service listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
