from __future__ import annotations

from customer_service_tools.mcp_server import MCPServer, PROTOCOL_VERSION


class FakeToolService:
    def execute(self, name, arguments, context=None):
        return {
            "schema_version": "1.0",
            "tool": name,
            "request_id": "r1",
            "trace_id": "t1",
            "elapsed_ms": 1.0,
            "data": {"answer": "ok"},
        }


def test_initialize_and_realtime_exposure() -> None:
    server = MCPServer(FakeToolService(), exposure="realtime")
    initialized = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION}}
    )
    assert initialized["result"]["protocolVersion"] == PROTOCOL_VERSION
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["answer_customer_question"]


def test_tool_call_returns_text_and_structured_content() -> None:
    server = MCPServer(FakeToolService(), exposure="realtime")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "answer_customer_question", "arguments": {"question": "hello"}},
        }
    )
    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["data"]["answer"] == "ok"
    assert result["content"][0]["type"] == "text"


def test_admin_tool_is_not_available_in_realtime_profile() -> None:
    server = MCPServer(FakeToolService(), exposure="realtime")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "publish_knowledge_version", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True
    assert response["result"]["structuredContent"]["error_code"] == "permission_denied"
