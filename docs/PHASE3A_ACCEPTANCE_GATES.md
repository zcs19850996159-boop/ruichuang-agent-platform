# Phase 3A acceptance gates

## Compatibility

- Competition `/chat` contract smoke test passes unchanged.
- Phase 1 and Phase 2 automated tests remain green.
- Frozen 400-question regression metrics must not regress.
- Real-time requests do not call the control-plane database after an identity
  has been resolved except for token status and tenant membership validation.

## Authentication and credentials

- Control API is disabled unless explicitly enabled.
- Bootstrap is disabled without a configured bootstrap token.
- Enterprise API tokens contain at least 256 bits of randomness.
- Plaintext API tokens are returned once and never stored.
- Invalid, expired, or revoked tokens return HTTP 401.

## Tenant isolation

- Tenant identity comes from the authenticated principal, not `X-Tenant-Id`.
- A token for tenant A cannot read or mutate tenant B.
- Managed images, knowledge spaces, active pointers, caches, and audit queries
  remain tenant-scoped.
- A cross-tenant request returns HTTP 403 without confirming whether the target
  resource exists.

## RBAC

- `owner` and `admin` can manage members and API tokens.
- `knowledge_manager` can ingest, evaluate, publish, and roll back knowledge.
- `agent` can answer questions and read published knowledge only.
- `viewer` can read knowledge metadata and audit records only.
- Every privileged mutation writes an audit event with tenant, actor, action,
  resource, trace, and outcome.

## Phase 3A exclusions

- Password login, SSO, SCIM, billing, quotas, task queues, object storage,
  PostgreSQL, and a graphical administration console.
- A general-purpose cloud WorkBuddy implementation.
