# ADR 0009: Production persistence uses PostgreSQL and immutable object storage

Status: accepted for Phase 3C

## Decision

The enterprise control plane uses PostgreSQL through a bounded connection pool
and explicit versioned migrations. SQLite remains supported only for local
development, deterministic tests, and single-node acceptance.

Published knowledge packages use S3-compatible object storage. Local disk is a
staging and read-cache layer, not the durable source of truth. The manifest is
written last, so its presence marks a complete immutable version. The active
pointer is a small replaceable object; version contents are never overwritten.

## Consequences

- A clean node can reconstruct the active knowledge cache.
- Database and object storage can be backed up with provider-native tooling.
- Production startup requires both durable services and fails closed when the
  deployment tries to fall back to SQLite or local-only knowledge.
- The real-time answer path reads local cached knowledge and does not add an S3
  round trip after a version has been materialized.
