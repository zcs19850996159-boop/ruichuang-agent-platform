# Phase 3C 3.5.3 production deployment

This directory is the source-controlled snapshot of the active production
deployment. It intentionally contains no credentials, generated certificates,
data, logs, PID files, or downloaded service binaries.

Layout:

- `root/start_ruichuang.sh`: complete service startup and readiness gate;
- `etc/autodl.sh`: host boot entrypoint;
- `etc/nginx/`: public port 6006 and shared proxy configuration;
- `etc/logrotate.d/`: Phase 3C log policy;
- `infra/scripts/`: infrastructure, KES setup, object-store policy, guardian,
  health monitoring, least-privilege application identity, and daily recovery
  scripts;
- `infra/policies/`: the bucket-prefix-limited Phase 3 application policy;
- `infra/supervisor/`: supervisor process definition;
- `KES_BINARY.sha256`: pinned official KES release checksum.

The runtime secrets remain at
`/root/autodl-tmp/customer_agent_phase3/infra/secrets/` with mode 0600. KES
certificate/key material remains under `infra/kes/` with root-only permissions.
Restore those files from a protected recovery generation; never regenerate
them for an existing encrypted MinIO data directory.

The host must provide PostgreSQL 14 client/server tools, Redis, Nginx,
Supervisor, `logrotate` (3.19 or newer), and the MinIO client. Daily
maintenance fails visibly at startup if `logrotate` is missing instead of
silently skipping rotation.

The API process uses `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, which must
belong to the `ruichuang-phase3c-app` policy and must not equal the MinIO root
identity. MinIO root variables are removed from the API process environment.
The active `ruichuang-phase3c-locked` bucket was created with Object Lock.
All 287 current objects and all 584 historical versions were migrated and
verified, and receive 30-day GOVERNANCE retention. The original bucket remains
unchanged as a migration rollback source.

Release 3.5.3 retains concurrency slots until non-cancellable worker threads
actually finish, includes queue time in request deadlines, records SSE failure
outcomes, makes final-owner protection transactional, rejects token issuance
for inactive identities, enforces the 16 MiB Tool API JSON limit in both Nginx
and FastAPI, and verifies the connected remote-media peer after DNS resolution.
