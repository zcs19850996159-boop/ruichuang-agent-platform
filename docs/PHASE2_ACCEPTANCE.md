# Phase 2 Acceptance

Status: passed. Container image digest remains a CI/build-host deliverable because
Docker/BuildKit is unavailable on the validation host.

## Delivered

- Tool API schema `1.0` shared by REST, MCP, and WorkBuddy.
- The real-time MCP profile exposes only `answer_customer_question`.
- WorkBuddy displays the domain answer verbatim and does not create a second Agent loop.
- Diagnostic evidence search, image identification, and deterministic answer validation.
- Administrative manual ingestion, quality evaluation, immutable publish, atomic
  activation, rollback, and batch answer audit.
- Text, Markdown, DOCX, native PDF, and Tesseract-backed scanned PDF ingestion.
- MIME/content validation, size limits, optional malware scanning, prompt-injection
  quarantine, OCR quality blocking, regression gates, and separate admin credentials.
- Page-aware PDF image extraction/binding and authenticated published-image delivery.

## Verification

- Phase 1 and Phase 2 tests: 20/20 passed.
- Official API contract smoke test: passed.
- Frozen 400-question regression after isolated nondeterministic verifier retries:
  API, route, manual, constraints, and PIC/image-count checks 100%.
- Image exact set/order: 99.5%; image F1: 0.9991, equal to Phase 1.
- Warm-cache HTTP P50, 20 requests per endpoint:
  `/chat` 13.608 ms; Tool API 14.236 ms; adapter delta 0.629 ms.
- Real admin integration: unauthorized request rejected; staging, evaluation,
  v1 publish, v2 publish, search, and rollback to v1 passed.

## Host integration acceptance

- Local Codex project Skill: passed.
- Local stdio MCP proxy handshake and one-tool exposure: passed.
- Real Codex host invocation through the SSH tunnel and REST Tool API: passed.
- The host called `answer_customer_question` exactly once and returned
  `data.answer` verbatim.
- Real multimodal URL input through Codex: passed. The proxy securely downloaded
  the public image, blocked private/local address targets, converted it to a
  bounded `data:image/...;base64` attachment, and the service returned the
  expected `Manual01_18` image.
- Measured real request: Tool API `12,878.36 ms`; proxy/tunnel end-to-end
  `13.200 s`. The host adapter added about `0.322 s`; current latency is
  dominated by the customer-service core/model path.
- Retrieval cold start is now performed during application startup. Measured
  startup warmup was `10,374.6 ms`; health reports the warmup state.
- Warm repeated question: about `164 ms` wall time and `28 ms` server time.
- Warm uncached deterministic-route question: about `3.51 s`; a question that
  invokes the LLM image selector remained about `7.45 s`.
- A full Codex Agent turn is not an end-customer real-time path: the acceptance
  environment spent more than a minute retrying host-model sampling before its
  single tool call. Codex remains suitable as a staff/admin host over the
  dedicated real-time Tool API.
- The server API token is stored in the macOS login Keychain under the
  `ruichuang-customer-service-api-token` service and is no longer present in
  project configuration. The MCP proxy reads it at request time.
- A dedicated Ed25519 SSH key is installed with a forced-failure command and
  restrictions that permit forwarding only to server `127.0.0.1:8877`.
  Acceptance confirmed that the approved tunnel succeeds, remote shell access
  fails, and forwarding to another server port fails.
- Server API authentication is required again. An unauthenticated Tool API call
  returned HTTP 401, while an authenticated MCP call using the Keychain
  credential returned a grounded answer.
- Local and remote temporary token-transfer files were deleted after the
  Keychain import.
- WorkBuddy host acceptance: pending.

## Boundaries

- Competition `/chat` and frozen knowledge assets remain unchanged.
- Managed enterprise knowledge is a separate immutable store and is selected only
  by enterprise tenant/knowledge-space context.
- Multi-tenant cloud control plane, billing, organization RBAC, and task queues remain
  Phase 3 scope.
