# Phase 3C acceptance gates

Phase 3C converts the accepted single-node enterprise slice into production
infrastructure. It remains isolated from the Phase 2 service until every gate
below passes.

## C1: persistence and migrations

- The control service depends on a store protocol, not a SQLite class.
- SQLite development databases migrate from Phase 3A/3B to schema v3 without
  data loss.
- A database newer than the application fails closed.
- PostgreSQL uses a bounded connection pool and a transaction advisory lock
  during repeatable migrations.
- Production mode refuses SQLite.
- Health output exposes only backend and schema state, never a DSN or path.

## C2: tenant object storage

- Only approved immutable knowledge releases are copied to object storage.
- Staging and document parsing remain in a bounded local work area.
- Every object key is rooted beneath its tenant and knowledge-space scope.
- Unsafe or traversal-like keys are rejected.
- A clean application node can restore the active pointer and materialize a
  published version into its local read cache.
- Publishing an existing version cannot replace different bytes.
- Object-store credentials, bucket names, endpoints, and physical paths are
  absent from public APIs.

## C3: production configuration

- Production startup requires PostgreSQL, S3-compatible object storage, Redis,
  explicit CORS origins, authentication, strong secrets, and explicit trusted
  proxy IPs.
- Placeholder secrets, wildcard CORS, debug mode, or spoofable proxy trust make
  startup fail closed.
- `/live` checks process liveness without dependency calls.
- `/ready` checks retrieval, control-plane persistence, and managed knowledge.
- Security headers are set globally; HSTS is enabled only in production.

## Compatibility and deployment

- All Phase 1, Phase 2, Phase 3A, and Phase 3B tests remain green.
- Object-storage tests prove cross-tenant isolation and clean-node recovery.
- A real PostgreSQL migration and CRUD run must pass before any production
  switch.
- A real S3-compatible publish/read/rollback run must pass before any
  production switch.
- `/chat` and the single-call real-time Tool API retain their existing
  contracts and latency path.
- The Phase 2 service on port 8877 is not modified during Phase 3C acceptance.
