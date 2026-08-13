# ADR-0008: derive enterprise tenant context from authenticated identity

Status: accepted.

Enterprise HTTP callers must not select a tenant by sending `X-Tenant-Id`.
The control plane authenticates the bearer token, resolves its tenant,
membership, role, and permissions, and overwrites untrusted identity headers
before invoking customer-service tools.

The legacy competition credential remains mapped to the frozen
`default/competition` scope. This preserves the official `/chat` contract while
preventing enterprise tokens from crossing tenant boundaries.

Phase 3A uses a local SQLite control store for deterministic development and
acceptance. All access is behind `ControlPlaneService` so Phase 3C can replace
the persistence implementation with PostgreSQL without changing the
customer-service core.
