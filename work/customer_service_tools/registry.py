from __future__ import annotations

from typing import Any


BASE_PROPERTIES = {
    "schema_version": {"type": "string", "const": "1.0", "default": "1.0"},
    "request_id": {"type": "string"},
}
OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "schema_version": {"type": "string", "const": "1.0"},
        "tool": {"type": "string"},
        "request_id": {"type": "string"},
        "trace_id": {"type": "string"},
        "elapsed_ms": {"type": "number"},
        "data": {"type": "object"},
    },
    "required": ["schema_version", "tool", "request_id", "trace_id", "elapsed_ms", "data"],
    "additionalProperties": False,
}


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {**BASE_PROPERTIES, **properties},
        "required": required,
        "additionalProperties": True,
    }


TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "answer_customer_question": {
        "description": "Answer one real-time text or multimodal customer-service question. Call exactly once per ordinary user question and display the returned answer verbatim.",
        "audience": "realtime",
        "required_permission": "customer:answer",
        "inputSchema": _schema(
            {
                "question": {"type": "string", "minLength": 1},
                "attachments": {"type": "array", "items": {"type": ["string", "object"]}, "maxItems": 8},
                "conversation_context": {"type": "object"},
                "response_mode": {
                    "type": "string",
                    "enum": ["sync", "stream"],
                    "default": "sync",
                },
            },
            ["question"],
        ),
    },
    "search_customer_evidence": {
        "description": "Retrieve customer-service evidence without composing a final answer.",
        "audience": "diagnostic",
        "required_permission": "knowledge:read",
        "inputSchema": _schema(
            {
                "question": {"type": "string", "minLength": 1},
                "manual_hint": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            ["question"],
        ),
    },
    "identify_product_image": {
        "description": "Analyze supplied product images and return structured grounding; does not answer the user.",
        "audience": "diagnostic",
        "required_permission": "customer:answer",
        "inputSchema": _schema(
            {
                "question": {"type": "string"},
                "attachments": {"type": "array", "items": {"type": ["string", "object"]}, "minItems": 1, "maxItems": 8},
            },
            ["attachments"],
        ),
    },
    "validate_customer_answer": {
        "description": "Run deterministic format and evidence checks on a proposed answer.",
        "audience": "diagnostic",
        "required_permission": "knowledge:read",
        "inputSchema": _schema(
            {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "images": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "object"}},
            },
            ["answer"],
        ),
    },
    "ingest_customer_manual": {
        "description": "Securely inspect and stage a new manual; never publishes it directly.",
        "audience": "admin",
        "required_permission": "knowledge:write",
        "inputSchema": _schema(
            {
                "source_path": {"type": "string", "minLength": 1},
                "product_id": {"type": "string", "minLength": 1},
            },
            ["source_path", "product_id"],
        ),
    },
    "evaluate_knowledge_update": {
        "description": "Evaluate a staged manual package and produce a publish gate report.",
        "audience": "admin",
        "required_permission": "knowledge:write",
        "inputSchema": _schema(
            {
                "staging_id": {"type": "string"},
                "regression_report": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "total": {"type": "integer", "minimum": 0},
                        "passed": {"type": "integer", "minimum": 0},
                        "failed": {"type": "integer", "minimum": 0},
                    },
                },
            },
            ["staging_id"],
        ),
    },
    "publish_knowledge_version": {
        "description": "Publish an approved immutable knowledge version and atomically activate it.",
        "audience": "admin",
        "required_permission": "knowledge:publish",
        "inputSchema": _schema(
            {
                "staging_id": {"type": "string"},
                "version": {"type": "string"},
                "approved_by": {"type": "string"},
            },
            ["staging_id", "version", "approved_by"],
        ),
    },
    "rollback_knowledge_version": {
        "description": "Atomically switch the active pointer to an existing immutable version.",
        "audience": "admin",
        "required_permission": "knowledge:publish",
        "inputSchema": _schema({"target_version": {"type": "string"}}, ["target_version"]),
    },
    "audit_customer_answers": {
        "description": "Batch-run deterministic customer-answer checks for operations and regression review.",
        "audience": "admin",
        "required_permission": "audit:read",
        "inputSchema": _schema(
            {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1000,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": ["string", "integer"]},
                            "answer": {"type": "string"},
                            "images": {"type": "array", "items": {"type": "string"}},
                            "evidence": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": ["answer"],
                    },
                }
            },
            ["items"],
        ),
    },
}


def list_tools(profile: str = "realtime") -> list[dict[str, Any]]:
    allowed = {
        "realtime": {"realtime"},
        "diagnostic": {"realtime", "diagnostic"},
        "admin": {"realtime", "diagnostic", "admin"},
    }.get(profile, {"realtime"})
    return [
        {
            "name": name,
            "description": definition["description"],
            "inputSchema": definition["inputSchema"],
            "outputSchema": OUTPUT_SCHEMA,
        }
        for name, definition in TOOL_DEFINITIONS.items()
        if definition["audience"] in allowed
    ]
