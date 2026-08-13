# ADR 0006: Real-time clients make one domain-tool call

Status: accepted

Ordinary customer questions from WorkBuddy, MCP clients, or another Agent call only
`answer_customer_question`. The client displays the domain answer verbatim and does
not create its own retrieve/analyze/validate loop.

The customer-service core retains deterministic routing, retrieval, multimodal
grounding, generation, and validation. This preserves the existing latency budget,
behavioral regression suite, and evidence discipline.

Diagnostic tools exist for operators. Knowledge administration tools are isolated
behind separate credentials and are not exposed by the real-time MCP profile.
