# Phase 3A progress

Status: first control-plane slice passed on the isolated Phase 3 server copy.
The running Phase 2 service was not replaced.

## Delivered

- Independent `customer_agent_phase3` baseline copied from the accepted Phase 2
  application.
- SQLite development store for tenants, users, memberships, API tokens,
  knowledge spaces, and append-only audit events.
- `owner`, `admin`, `knowledge_manager`, `agent`, and `viewer` RBAC roles.
- 256-bit random enterprise API tokens persisted only as HMAC-SHA256 hashes.
- Token expiration and revocation enforcement.
- Bootstrap, identity, member, token, knowledge-space, and audit REST endpoints.
- Enterprise Tool API identity binding. Tenant, user, role, and permissions are
  derived from the authenticated principal instead of untrusted headers.
- Knowledge-space existence and permission checks occur before the
  customer-service core is called.
- Successful enterprise knowledge-administration tool calls are recorded in the
  central control-plane audit log.
- Cloud control is disabled by default and fails closed when its token pepper or
  bootstrap credential is missing.

## Verification

- Phase 3 tests: 9/9 passed.
- Phase 1 and Phase 2 tests on the Phase 3 codebase: 20/20 passed.
- Real HTTP acceptance on isolated server port:
  - tenant A bootstrap: HTTP 201;
  - tenant B bootstrap: HTTP 201;
  - knowledge-space creation: HTTP 201;
  - member creation: HTTP 201;
  - token issue: HTTP 201;
  - missing identity: HTTP 401;
  - tenant A reading tenant B: HTTP 403;
  - `agent` writing knowledge-space configuration: HTTP 403;
  - `agent` reading its tenant spaces: HTTP 200;
  - tenant-scoped audit query: HTTP 200.
- Database inspection confirmed fixed-length token hashes and no plaintext token
  columns.
- Legacy authenticated `/chat` returned HTTP 200 with a non-empty answer and the
  Competition Profile.
- The Phase 3 acceptance process and temporary database/credentials were
  removed after testing.
- The existing Phase 2 service remained alive throughout acceptance.

## Next slice

Phase 3B should replace acceptance-only provisioning with a usable
administration workflow:

1. PostgreSQL persistence interface and migrations.
2. Organization/member lifecycle, role changes, and user/token suspension.
3. Knowledge-space list/version/publish/rollback control endpoints.
4. Minimal authenticated Web administration console.
5. Tenant-isolated object-storage adapter and backup policy.
6. Concurrency, cache-isolation, and negative authorization tests.
