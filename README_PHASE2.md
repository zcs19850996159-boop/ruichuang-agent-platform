# Phase 2: WorkBuddy/MCP adapters and managed knowledge

Phase 2 keeps the competition `/chat` contract and frozen knowledge assets intact.
It adds a single versioned domain-tool layer used by REST, MCP, and WorkBuddy.

## Real-time tool

`POST /tools/v1/answer_customer_question`

```json
{
  "schema_version": "1.0",
  "question": "如何安装电池？",
  "attachments": [],
  "response_mode": "sync"
}
```

WorkBuddy must call this tool exactly once for an ordinary customer question and
display `data.answer` without another model rewrite.

## Knowledge lifecycle

Administrative tools require both the normal API credential and
`X-Admin-Token`, matched against `KNOWLEDGE_ADMIN_TOKEN`.

1. `ingest_customer_manual`
2. `evaluate_knowledge_update`
3. human review
4. `publish_knowledge_version`
5. `rollback_knowledge_version` if required

Uploaded document content is treated as untrusted evidence. Publication is blocked
on content/type mismatch, detected prompt injection, empty extraction, or a scanned
PDF without a configured OCR provider. Set `KNOWLEDGE_REQUIRE_REGRESSION=1` in
strict production environments to require a regression report before publication.
The container includes Tesseract Chinese/English OCR; enable it with
`KNOWLEDGE_OCR_PROVIDER=tesseract`.

Published knowledge versions are immutable. Activation and rollback atomically
replace only the active pointer for a tenant and knowledge space.

## MCP

Run with the project `work/` directory on `PYTHONPATH`:

```bash
CUSTOMER_MCP_EXPOSURE=realtime python -m customer_service_tools.mcp_server
```

The default profile exposes only the real-time answer tool. Admin exposure requires
both `CUSTOMER_MCP_EXPOSURE=admin` and `CUSTOMER_MCP_ALLOW_ADMIN=1`.
