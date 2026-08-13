# Phase 3C production operations

## Current production release

The active release is **Phase 3C 3.5.3**, served by FastAPI on loopback port
`8878` and published through Nginx port `6006`. Phase 2 on `8877` is retained
only as a rollback service and is not part of the active public route.

The authoritative deployment snapshot lives in
`deploy/phase3c-production/`. The production baseline branch is
`phase3c-production-3.5.3` and the matching immutable tag is
`phase3c-production-v3.5.3`. Do not deploy from the older Phase 2 branch or
from historical deployment examples preserved in the recovery archive.

Supervisor manages four long-running production controls:

- KES on TLS-protected loopback port `7373`;
- the service guardian, which validates every required dependency and performs
  bounded recovery;
- daily snapshots and log rotation;
- a local health monitor that writes
`infra/run/health-monitor.json`. Optional external alert delivery is enabled by
placing `ALERT_WEBHOOK_URL` in root-only `infra/secrets/monitoring.env`; without
that file, monitoring remains local and must not be described as external
alerting.

The host-level `logrotate` package is a required runtime dependency. The
maintenance process refuses to start if it is unavailable.

MinIO uses KES-backed SSE-S3. The primary bucket has versioning, AES256 default
encryption, 30-day GOVERNANCE retention for new objects, and lifecycle
retention for noncurrent versions. The locked bucket was populated by copying
and verifying all 287 current objects and all 584 historical versions. The
original bucket remains unchanged as a migration rollback source. The KES recovery
material is root-only and is included in daily recovery snapshots; losing it
would make encrypted object data unrecoverable.

Release 3.5.3 adds the following runtime invariants:

- a timed-out or disconnected request keeps its concurrency slot until the
  underlying worker thread has actually returned;
- the request deadline includes time spent waiting for a concurrency slot;
- streaming success, input failure, timeout, internal error, and client
  disconnect outcomes are written to the structured metrics log;
- the final active tenant owner cannot be removed by concurrent transactions;
- new tokens are issued only for active tenant, user, and membership records;
- Tool API and control-plane JSON bodies are capped at 16 MiB in the
  application, with a matching Tool API Nginx limit;
- remote-media redirects are revalidated and the connected peer IP must still
  be globally routable. Environment HTTP proxies are ignored by that resolver.

## Deployment boundary

The public ingress terminates TLS and forwards only to the FastAPI service on a
private address. PostgreSQL, Redis, and S3-compatible storage are never exposed
to the public network.

The public ingress returns HTTP 404 for `/docs`, `/redoc`, and
`/openapi.json`. Operators may use `http://127.0.0.1:8878/docs` from the server
when schema inspection is required.

Required production settings are validated at application startup. The process
refuses to start when it detects SQLite, local-only knowledge storage, wildcard
CORS, placeholder secrets, debug mode, or unrestricted proxy-header trust.

Use:

- `/live` for process liveness;
- `/ready` for load-balancer readiness;
- `/health` for authenticated operational inspection;
- `/metrics` for the existing authenticated structured counters.

The ingress source addresses must exactly match `API_TRUSTED_PROXY_IPS`.
Forwarded headers from any other peer are ignored.

## Database migrations

Control-plane migrations execute before the application accepts traffic.
PostgreSQL migration workers serialize through a transaction advisory lock.
Deploy one compatible application version at a time and verify `/ready` before
removing the previous version.

Never roll an older application onto a newer schema. It fails closed by design.

## Backups

For local acceptance only, the operational CLI can create online SQLite
snapshots and knowledge archives:

```bash
PYTHONPATH=work python work/phase3_operations.py sqlite-backup \
  --source outputs/control/control-plane.sqlite3 \
  --output backups/control-plane.sqlite3

PYTHONPATH=work python work/phase3_operations.py knowledge-backup \
  --source knowledge_store \
  --output backups/knowledge-store.tar.gz
```

Staging files are intentionally excluded from knowledge archives.

This deployment creates a root-only daily recovery generation under
`/root/autodl-tmp/customer_agent_phase3/backups/daily/`. Each complete
generation contains a PostgreSQL custom-format dump, a MinIO data archive,
Phase 3 and exact Phase 2 rollback Git bundles, KES recovery material,
operational configuration, and SHA-256 checksums. Keep an independent off-host
copy to protect against host loss.

The production API receives only the prefix-limited `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. MinIO root credentials are retained for infrastructure
recovery but are explicitly removed from the API process environment. The
startup gate refuses production launch when the two identities are equal.

Production object storage must enable:

- bucket versioning;
- server-side encryption;
- lifecycle retention;
- restricted service identities;
- cross-region or independent-account replication when required;
- deletion protection for immutable version prefixes.

## Restore drill

Local restore commands require the exact confirmation phrase
`RESTORE_RUICHUANG_DATA`. When replacing existing data, the tools preserve a
pre-restore recovery copy.

Every quarterly production drill must prove:

1. restore PostgreSQL to an isolated environment;
2. restore or mount a consistent object-storage generation;
3. start the matching application version;
4. verify tenant/member/token state;
5. verify active knowledge pointers and random object checksums;
6. run cross-tenant isolation tests;
7. run one managed-knowledge answer and one legacy `/chat` request;
8. record actual RPO and RTO.

No restored environment may send external customer traffic until the drill is
complete.
