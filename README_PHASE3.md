# Phase 3: Enterprise cloud control plane

Phase 3 adds a tenant-aware control plane around the frozen customer-service
data plane. It does not replace the competition `/chat` path and does not put a
general Agent loop in front of real-time questions.

## Phase 3A scope

- Tenant, user, membership, role, API-token, and knowledge-space records.
- High-entropy API tokens stored only as keyed hashes.
- Identity-derived tenant context; enterprise callers cannot choose another
  tenant through request headers.
- RBAC for customer answering, knowledge administration, publishing, user
  administration, and audit access.
- Append-only control-plane audit events.
- SQLite development/acceptance store behind a service boundary. PostgreSQL is
  the production-store target for Phase 3C.

## Explicitly unchanged

- `POST /chat` competition behavior and frozen knowledge assets.
- `POST /tools/v1/answer_customer_question` single-call real-time behavior.
- Immutable managed-knowledge versions and atomic activation/rollback.

## Phase 3A API

The control API is available only when `CLOUD_CONTROL_ENABLED=1`.

- `POST /control/v1/bootstrap`
- `GET /control/v1/me`
- `POST /control/v1/tenants/{tenant_id}/members`
- `GET /control/v1/tenants/{tenant_id}/members`
- `PATCH /control/v1/tenants/{tenant_id}/members/{user_id}`
- `POST /control/v1/tenants/{tenant_id}/tokens`
- `GET /control/v1/tenants/{tenant_id}/tokens`
- `DELETE /control/v1/tenants/{tenant_id}/tokens/{token_id}`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces`
- `GET /control/v1/tenants/{tenant_id}/knowledge-spaces`
- `GET /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/versions`
- `GET /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/active`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/ingestions`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/package-ingestions`
- `GET /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/evaluate`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/regression`
- `GET /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/diagnosis`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/staging/{staging_id}/publish`
- `POST /control/v1/tenants/{tenant_id}/knowledge-spaces/{space_id}/versions/{version}/rollback`
- `GET /control/v1/tenants/{tenant_id}/audit`

Bootstrap additionally requires `X-Control-Bootstrap-Token`. Issued API tokens
are returned once and persisted only as hashes.

When cloud control is enabled, the minimal administration console is available
at `/admin`. It keeps the supplied API token only in page memory.

The HTTP `/evaluate` path is retained for API compatibility, but its
user-facing meaning is **ingestion quality check**, not competition answer
evaluation. Codex and WorkBuddy use the separate
`ruichuang-knowledge-operations` Skill to analyze authorized product folders,
detect file/version/model/policy conflicts, stage multi-document packages,
diagnose blockers, run knowledge-version regression, coordinate explicit
approval, publish, verify, and roll back through these tenant-scoped APIs. The
platform remains responsible for canonical parsing, chunking, image binding,
per-version hybrid indexing, and online RAG behavior.
