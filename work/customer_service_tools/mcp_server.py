from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import replace
from typing import Any, TextIO

from customer_service_core import RequestContextFactory
from customer_service_core.errors import CustomerServiceError

from .registry import TOOL_DEFINITIONS, list_tools
from .service import CustomerServiceToolService, create_tool_service


PROTOCOL_VERSION = "2025-11-25"


class MCPServer:
    """Minimal stable MCP stdio adapter; business logic remains in ToolService."""

    def __init__(self, service: CustomerServiceToolService, exposure: str = "realtime") -> None:
        self.service = service
        self.exposure = exposure if exposure in {"realtime", "diagnostic", "admin"} else "realtime"
        self.allowed = {tool["name"] for tool in list_tools(self.exposure)}

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = str(request.get("method") or "")
        request_id = request.get("id")
        if request_id is None and method.startswith("notifications/"):
            return None
        try:
            if request.get("jsonrpc") != "2.0":
                return self._error(request_id, -32600, "Invalid Request")
            if method == "initialize":
                return self._result(
                    request_id,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "ruichuang-customer-service",
                            "version": "3.5.0-phase3c",
                        },
                    },
                )
            if method == "ping":
                return self._result(request_id, {})
            if method == "tools/list":
                return self._result(request_id, {"tools": list_tools(self.exposure)})
            if method == "tools/call":
                params = request.get("params") or {}
                name = str(params.get("name") or "")
                if name not in self.allowed:
                    return self._tool_error(request_id, "permission_denied", "tool is not exposed by this MCP profile")
                arguments = params.get("arguments") or {}
                context = RequestContextFactory.from_request(payload=arguments)
                context = replace(
                    context,
                    profile=os.environ.get("CUSTOMER_MCP_CUSTOMER_PROFILE", context.profile),
                    tenant_id=os.environ.get("CUSTOMER_MCP_TENANT_ID", context.tenant_id),
                    knowledge_space_id=os.environ.get("CUSTOMER_MCP_KNOWLEDGE_SPACE_ID", context.knowledge_space_id),
                    role="admin" if self.exposure == "admin" else "user",
                    permissions=("knowledge:write",) if self.exposure == "admin" else (),
                )
                output = self.service.execute(name, arguments, context=context)
                return self._result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}],
                        "structuredContent": output,
                        "isError": False,
                    },
                )
            return self._error(request_id, -32601, "Method not found")
        except CustomerServiceError as exc:
            return self._tool_error(request_id, exc.code.value, exc.message)
        except (TypeError, ValueError) as exc:
            return self._tool_error(request_id, "input_invalid", str(exc))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return self._tool_error(request_id, "internal_error", str(exc))

    @staticmethod
    def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def _tool_error(self, request_id: Any, code: str, message: str) -> dict[str, Any]:
        payload = {"error_code": code, "message": message}
        return self._result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": True,
            },
        )


def serve_stdio(
    server: MCPServer,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> None:
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = server.handle(request)
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if response is not None:
            output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_stream.flush()


def main() -> None:
    exposure = os.environ.get("CUSTOMER_MCP_EXPOSURE", "realtime").strip().lower()
    if exposure == "admin" and os.environ.get("CUSTOMER_MCP_ALLOW_ADMIN", "0") != "1":
        raise SystemExit("admin MCP exposure requires CUSTOMER_MCP_ALLOW_ADMIN=1")
    serve_stdio(MCPServer(create_tool_service(), exposure=exposure))


if __name__ == "__main__":
    main()
