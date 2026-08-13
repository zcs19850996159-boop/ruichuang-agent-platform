# Phase 3C progress

## Production hardening completed (2026-08-04, release 3.5.2)

- Active application version unified at `3.5.2`; Phase 2 remains rollback-only
  and is frozen at tag `phase2-rollback-warmup-v3.2.1`.
- PostgreSQL, MinIO, application, configuration, and KES recovery snapshots are
  generated with SHA-256 verification.
- Supervisor, boot recovery, dependency guardian, Redis AOF/RDB persistence,
  and daily maintenance are active and have passed a real reboot recovery.
- Public URL discovery, trusted proxy handling, request-body limits,
  authenticated detailed health, and public sanitized readiness are enforced.
- MinIO uses KES-backed server-side encryption; all 287 current objects were
  migrated and verified as AES256-encrypted. Versioning and noncurrent-version
  lifecycle retention are enabled.
- The application uses a dedicated prefix-limited MinIO identity rather than
  the MinIO root identity. The active Object-Lock-enabled bucket contains all
  287 current objects and all 584 historical versions, each migrated and
  verified with 30-day GOVERNANCE retention. The original bucket is retained
  unchanged as a migration rollback source.
- A local health monitor records dependency state and supports optional webhook
  delivery through a separate root-only `monitoring.env` file.
- Public interactive API documentation is blocked at Nginx and remains
  available only on the loopback application port.
- Daily/size-based log rotation is active. 168 untracked AppleDouble/backup
  artifacts were copied to the production hardening quarantine and removed
  from the live application tree.
- The complete isolated regression suite passed: 90 tests, with one upstream
  Starlette deprecation warning.
- The authoritative deployment snapshot is `deploy/phase3c-production/`; the
  production Git baseline is branch `phase3c-production-3.5.2`, tag
  `phase3c-production-v3.5.2`.

Status: the production foundation, real PostgreSQL/S3-compatible integration,
unified customer workbench, and Tool API streaming restoration have passed.

## Production bug fixes completed (2026-08-04, release 3.5.3)

- Blocking model work now retains its configured concurrency slot after an
  HTTP timeout or client disconnect until the worker thread actually exits.
  Queue wait is included in the end-to-end request deadline.
- Final-owner protection moved into the membership update transaction. On
  PostgreSQL the tenant membership rows are locked before the invariant is
  evaluated, preventing concurrent demotions from leaving zero owners.
- Token issuance locks and validates the target membership and rejects
  inactive tenant, user, or membership identities.
- Tool API and control-plane JSON requests now enforce the 16 MiB application
  limit before JSON expansion. Nginx applies the same limit to `/tools/v1`.
- Remote-media fetching ignores environment proxies and fails closed unless
  the actual connected peer remains a globally routable IP after DNS
  resolution and on every redirect.
- SSE failures, timeouts, and client disconnects are recorded in structured
  metrics; unexpected synchronous and streaming exceptions are logged without
  exposing their details to clients.
- Three silently shadowed duplicate function definitions were removed or
  explicitly renamed as legacy inline UI code.
- The isolated regression suite passes 97 tests. Candidate acceptance covered
  synchronized and streaming chat, an SSE error metric, the Manual27
  three-image answer, PostgreSQL/Redis/MinIO readiness, and a real HTTPS remote
  media fetch.
- The exact pre-fix rollback snapshot is
  `/root/autodl-tmp/customer_agent_phase3/backups/pre-bugfix-20260804-232225`.
- The production Git baseline is branch `phase3c-production-3.5.3`, tag
  `phase3c-production-v3.5.3`.

Phase 3 is active on loopback port 8878 for the controlled local Workbench and
Codex/MCP competition adapter. Phase 2 remains healthy on port 8877 as the
immediate rollback service. The AutoDL public ingress on port 6006 now proxies
to Phase 3 through Nginx; Phase 2 is not in the active public request path.

## Delivered

- A control-plane persistence protocol independent of SQLite.
- Explicit schema migrations through schema v3.
- Legacy Phase 3A/3B SQLite migration with data preservation.
- Fail-closed handling for databases newer than the application.
- PostgreSQL store implementation with a bounded synchronous pool and migration
  advisory lock.
- Environment-driven SQLite/PostgreSQL store selection and a production guard
  that rejects SQLite.
- S3-compatible immutable knowledge object-store adapter.
- Tenant/space/version object-key isolation and traversal rejection.
- Manifest-last publication and active-pointer storage.
- Clean-node materialization of immutable knowledge into a local read cache.
- Production configuration validation for database, object storage, Redis,
  authentication, secrets, CORS, debug mode, and trusted proxies.
- `/live` and `/ready` probes plus dependency state in `/health`.
- Trusted-proxy allowlisting before accepting `X-Forwarded-For`.
- Global security headers and production-only HSTS.
- Online SQLite snapshot, atomic restore with recovery copy, knowledge archive,
  and safe restore helpers for local acceptance/disaster-recovery exercises.
- Nginx TLS/reverse-proxy example and an operations runbook.
- Root-only infrastructure start, stop, and status scripts.
- Unified customer-service chat workbench at `/workbench`:
  - Competition Profile with in-memory legacy-token authentication and the
    frozen `competition` knowledge space;
  - Enterprise Profile with in-memory enterprise-token authentication and
    tenant knowledge-space selection;
  - one customer-service core and one versioned Tool API for both profiles;
  - server-side identity/profile/knowledge-space binding that ignores spoofed
    tenant, role, and space headers;
  - single-call `answer_customer_question` workflow;
  - text and up-to-three-image input;
  - evidence-image rendering;
  - knowledge version, confidence, latency, validation, evidence, and escalation
    trace;
  - responsive desktop/mobile layout and navigation to the administration
    console.
- Production startup now starts the local Qwen vision provider before Phase 3,
  and `/ready` fails closed when the configured provider is not actually loaded.
- Direct image questions prefer reviewed fingerprint grounding and accepted
  CLIP manual-image matches before generic model captions; low-confidence image
  clarifications are explicitly marked for human escalation.
- Restored streaming output through the same versioned Tool API endpoint:
  - `response_mode: "sync"` preserves the existing Tool API and MCP behavior;
  - `response_mode: "stream"` returns Server-Sent Events from the same
    `answer_customer_question` tool;
  - tenant, profile, role, and knowledge-space authorization completes before
    the stream opens;
  - native model deltas are forwarded without a second customer-core call or a
    second model rewrite;
  - accepted/status, answer metadata, answer delta/reset, and final events are
    rendered incrementally by `/workbench`;
  - every completed `<PIC>` marker inserts its corresponding secured manual
    image immediately at that position, before later answer text is generated;
  - image blobs are fetched once and retained across subsequent delta renders,
    avoiding image flicker and repeated authenticated downloads;
  - a paced fallback is available when the underlying response is already
    cached and therefore has no native deltas.
- Competition image compatibility that converts legacy image IDs into secured
  `/manual-images/{image_id}` requests.
- Manual27 battery-install topic normalization: generic "更换电池" requests now
  resolve to the three battery-install figures instead of battery-status
  figures.
- Safe escalation when an enterprise knowledge space has no active published
  version instead of exposing a technical storage error.
- Automated official TXT ingestion for the competition-provided
  `[manual text, image id list]` envelope:
  - manual text is extracted from the envelope instead of indexing raw JSON;
  - ordered `<PIC>` markers are bound to the ordered official image IDs;
  - referenced images are copied into the immutable managed version;
  - missing referenced images block publication.
- Aggregate knowledge updates:
  - a new product manual is appended to the active version;
  - a revised manual replaces only the matching product;
  - unrelated published manuals and images remain in the next version;
  - no manual chunk construction or index-building command is required from
    the operator.

## Server acceptance infrastructure

The following temporary acceptance infrastructure now exists on the server and
is bound to loopback only:

- PostgreSQL 14.23:
  - data: `/var/lib/postgresql/phase3c`;
  - listen address: `127.0.0.1:55432`;
  - database: `ruichuang_phase3c`;
  - credentials: root-only environment file outside the application tree.
- MinIO `RELEASE.2025-09-07T16-13-09Z`:
  - data: `/root/autodl-tmp/customer_agent_phase3/infra/minio-data`;
  - API: `127.0.0.1:59000`;
  - console: `127.0.0.1:59001`;
  - primary bucket: `ruichuang-phase3c`;
  - recovery bucket: `ruichuang-phase3c-restore`;
  - versioning enabled on both buckets.
- Operations scripts:
  `/root/autodl-tmp/customer_agent_phase3/infra/scripts/{start,stop,status}.sh`.

PostgreSQL, MinIO, and the loopback-only Phase 3 acceptance API remain running
for follow-up acceptance work.

## Verification

- Phase 3 tests: 43/43 passed.
- Phase 1 and Phase 2 compatibility tests: 29/29 passed.
- Final server regression: 72/72 passed, with one Starlette deprecation warning.
- Local smoke tests passed for:
  - schema v3 creation;
  - tenant bootstrap/authentication;
  - immutable object publication;
  - clean-cache rehydration;
  - cross-tenant object isolation;
  - SQLite snapshot and recovery;
  - knowledge archive and recovery.
- Local temporary HTTP server on `127.0.0.1:8878` passed with SQLite and
  filesystem storage:
  - `/live`: HTTP 200;
  - `/ready`: HTTP 200;
  - `/health`: HTTP 200 with SQLite schema v3 and filesystem knowledge status;
  - `/admin`: HTTP 200 with security headers;
  - tenant bootstrap and knowledge-space creation;
  - text manual ingest, evaluation, publication, and version inventory;
  - legacy competition `/chat`: HTTP 200 with a non-empty answer.
- Real PostgreSQL acceptance passed:
  - fresh schema v3 migration;
  - tenant, member, token, knowledge-space, and audit CRUD;
  - token authentication and revocation;
  - v2-to-v3 migration with data preservation;
  - 120 concurrent reads using 12 workers with a maximum pool size of 3;
  - stop/restart recovery through an existing connection pool;
  - logical backup, restore into a separate database, and data verification.
- Real MinIO acceptance passed:
  - immutable v1/v2 object publication;
  - clean-cache materialization and search;
  - rollback to v1;
  - cross-tenant isolation;
  - immutable-overwrite rejection;
  - versioning enabled on the primary and recovery buckets.
- Production-mode HTTP acceptance on temporary `127.0.0.1:8878` passed with
  PostgreSQL, MinIO, Redis, authentication, explicit CORS, trusted-proxy
  configuration, debug disabled, and HSTS:
  - `/ready`: HTTP 200 with PostgreSQL schema v3 and S3-compatible storage ready;
  - tenant bootstrap and knowledge-space creation;
  - cross-tenant access rejection (HTTP 403);
  - ingest, evaluate, and publish;
  - legacy competition `/chat`: HTTP 200 with a non-empty answer;
  - database audit events and an object-store manifest were verified.
- Recovery acceptance passed:
  - PostgreSQL logical backup restored into a separate database;
  - all 25 primary-bucket objects copied to the recovery bucket;
  - SHA-256 verification passed for every copied object;
  - a clean-cache service read and searched the active version from the
    recovery bucket;
  - cross-tenant reads remained empty.
- Existing Phase 2 service on `127.0.0.1:8877` remained healthy (HTTP 200).
- Browser workbench acceptance passed:
  - Competition/Enterprise mode switch in one `/workbench` route;
  - competition-token authentication through the same Tool API;
  - forced Competition Profile, `competition` knowledge space, and
    `competition-kb-v1` version;
  - "如何按照手册更换电池？" returned the complete three-step procedure,
    three matching manual figures, eight evidence items, and no escalation;
  - warm-cache workbench response completed in 25 ms after a 12.97-second cold
    generation;
  - enterprise owner authentication;
  - tenant knowledge-space discovery;
  - question submission through the Tool API;
  - evidence-insufficient safe escalation with no internal error leakage;
  - answer trace rendering;
  - desktop and responsive layouts;
  - external JavaScript/CSS under a strict Content Security Policy.
- Phase 3 Competition Profile cutover gate passed:
  - direct HTTP regression through the official-compatible `/chat` adapter;
  - first full run kept API, route, manual, PIC/image count, and image metrics
    equal to the frozen baseline;
  - three constraint-verifier variance rows (171, 244, 351) passed isolated
    retry;
  - the complete warm replay passed all 400 constraints with no issues;
  - image exact set/order remained 398/400 and average image F1 remained
    0.9991;
  - warm replay latency was P50 12.38 ms, P95 3.096 s, P99 4.238 s, maximum
    6.965 s;
  - normal rate limiting was restored after the regression.
- Startup retrieval warmup is enabled. It completed in 9.288 seconds before
  readiness and reduced the post-restart Manual27 battery smoke from the prior
  12.97-second observation to 2.515 seconds; the next exact-cache request
  completed in 4.01 milliseconds.
- A controlled local Competition Profile canary was executed on 2026-07-30:
  - the stable Codex/MCP endpoint remains `127.0.0.1:18877`;
  - its dedicated forwarding-only SSH key now permits only server
    `127.0.0.1:8878`, replacing the former `8877` target;
  - the local MCP and Skill configuration did not require an endpoint change;
  - Phase 3 was aligned to the frozen competition API credential, with the old
    Phase 3-only credential invalidated;
  - a real Skill request for "如何按照手册更换电池？" returned Manual27,
    `Manual27_1`/`Manual27_2`/`Manual27_3`, passed answer validation, did not
    escalate, and completed in 20.25 ms on an exact-cache hit;
  - Phase 3 `8878` and the retained Phase 2 rollback service on `8877` both
    remained ready after the switch.
- Tool API streaming restoration acceptance passed on 2026-07-30:
  - the same `POST /tools/v1/answer_customer_question` endpoint supports both
    sync JSON and SSE streaming responses;
  - the deployed native-model run emitted three status events, one answer
    metadata event, 72 answer-delta events, and one final event;
  - the first answer delta arrived in 789.13 ms and the complete response in
    2108.72 ms;
  - concatenated streamed text exactly matched both the final SSE payload and a
    separate compatibility sync response;
  - a fresh browser session rendered the Manual27 three-step battery procedure,
    `Manual27_1`, `Manual27_2`, and `Manual27_3`, with answer validation passed
    and no human escalation;
  - a mid-generation browser observation confirmed that the send control
    remained disabled while partial assistant text was already visible;
  - the local browser used the previously authorized Keychain credential only
    in page memory; the temporary transfer file and virtual clipboard value
    were cleared after authentication.
- Inline image streaming acceptance passed on 2026-07-30:
  - step 1 text became visible at 1369 ms;
  - `Manual27_1` appeared inline after step 1 at 1462 ms while steps 2 and 3
    were still absent and the response remained in progress;
  - step 2 text followed at 1553 ms;
  - `Manual27_2` appeared at 1644 ms and `Manual27_3` at 1735 ms;
  - the final response completed at 3532 ms with all three figures in marker
    order;
  - this verifies that the workbench no longer waits for the full text before
    inserting images.
- The repeatable Phase 3 workbench end-to-end gate passed on 2026-07-30:
  - its deliberately adversarial same-session scenario first asked a Manual17
    refrigerator-light question, then asked the explicit standalone question
    "如何按照手册更换电池？";
  - the gate exposed and fixed a real context-boundary defect in which a short
    explicit question could inherit the previous manual and return the wrong
    product answer;
  - after the fix, the second question resolved to Manual27 and rendered
    `Manual27_1`, `Manual27_2`, and `Manual27_3`, with validation passed and no
    escalation;
  - real browser observations saw only `Manual27_1` at 1553 ms, then
    `Manual27_1`/`Manual27_2` at 1847 ms, and all three images at 2056 ms while
    the response was still in progress; completion followed at 2850 ms;
  - the client now applies bounded image-paint backpressure so multiple SSE
    events arriving in one network read cannot collapse all inline figures into
    one browser paint;
  - new-chat isolation changed the session identifier, cleared prior messages,
    and restored the welcome state;
  - the server API gate emitted three status events, one metadata event, 72
    answer deltas, and one final event; concatenated deltas exactly equalled the
    final answer;
  - streaming Tool API, synchronous Tool API, and the official-compatible
    `/chat` adapter all selected Manual27 and the same ordered three-image set;
  - the retained Phase 2 rollback `/health` endpoint returned HTTP 200;
  - reusable browser, standard Playwright, and server API gate artifacts are
    stored under `work/phase3_remote/work/e2e/`.
- The legacy official TXT ingestion rehearsal passed on 2026-07-30:
  - an isolated enterprise tenant and `official-manuals` space were used, so
    the frozen Competition Profile was never used as an ingestion target;
  - `空气净化器手册.txt` was automatically parsed, chunked, image-bound,
    evaluated, and published as `official-txt-v1`;
  - `人体工学椅手册.txt` was then appended as `official-txt-v2`, retaining
    both documents rather than replacing the first manual;
  - the aggregate v2 contained two documents, four generated chunks, and 43
    copied manual images, with zero missing referenced images;
  - the enterprise answer for the chair lumbar-cushion massage function
    included the required USB instruction, passed validation, and did not
    escalate;
  - a published `Manual02_15` image was read through the authenticated image
    endpoint with HTTP 200;
  - rollback from v2 to v1 removed the chair evidence, and restoring v2 made it
    active again;
  - the same gate passed first on a non-traffic candidate and again after the
    controlled `8878` cutover;
  - the post-cutover streaming Tool API, synchronous Tool API, and official
    `/chat` adapter all retained the Manual27 three-image battery answer;
  - Phase 2 on `8877` and the local `18877` adapter both returned HTTP 200.
- Final-demo hardening passed on 2026-07-30:
  - `/admin` now supports browser manual upload, quarantined staging, visible
    quality metrics and blockers, explicit approval, immutable activation, and
    per-version rollback;
  - the upload endpoint enforces tenant, knowledge-space, staging ownership,
    file type, actual body size, and identity-derived approval, and preserves
    the user's original manual filename;
  - a persistent `final-demo / official-manuals` demonstration scope was
    created with `official-txt-v1` as its repeatable baseline;
  - the fixed chair-manual rehearsal produced two aggregate documents, four
    chunks, 27 available images, zero missing images, a grounded USB answer,
    no escalation, and a verified rollback to the baseline;
  - the fixed Competition cases select Manual27 with the ordered battery image
    set and Manual01 with `Manual01_18`; both passed validation without
    escalation after preheating;
  - the real browser logged into the enterprise console, selected the managed
    knowledge space, displayed versions and audits, and verified the release
    panel without leaving the hidden login panel in the page layout;
  - the real browser also rendered the Manual27 answer with all three inline
    images, Competition Profile, `competition-kb-v1`, evidence cards, and a
    passed answer check in 2025 ms;
  - `demo/final_demo_control.py` and the macOS launcher now provide strict
    `check`, `prepare`, `reset`, and `rollback` commands using Keychain
    credentials without printing tokens;
  - an independent rollback-only SSH key can forward only to server
    `127.0.0.1:8877`; the runtime switch script refuses to stop an unknown
    listener and completed a real Phase 3 to Phase 2 to Phase 3 round trip on
    stable local port `18877`;
  - the Phase 2 fallback returned its legacy `/ui` and official-compatible
    `/chat`; restoring Phase 3 returned `/workbench` and `/admin`;
  - the final server regression passed 77 tests with only the existing
    Starlette TestClient deprecation warning.

## Remaining external gates

The real integration environment is suitable for acceptance work, but it is
not a production object-storage or identity environment. The following must
not be reported as passed:

- provider-native PostgreSQL point-in-time recovery;
- independent-account or independent-region object-store recovery;
- OIDC/SSO and secure browser-session integration;
- production ingress certificate and external monitoring/alert delivery;
- An explicit timed rollback rehearsal for the current
  public Phase 3 ingress.
- Continued tail-latency monitoring under real customer concurrency. Normal
  requests now meet the 2–5 second target at P95, while a small cold-generation
  tail can still approach 7 seconds in the accepted warm replay.

KES-backed AES256, bucket versioning, lifecycle retention, a prefix-limited
application identity, and 30-day governance retention are now active. All 584
historical versions were copied into the Object-Lock-enabled active bucket and
verified; the original bucket remains intact for migration rollback. Until the
external gates are closed, the production configuration must continue to fail
closed where required. Phase 3 serves the controlled local adapter and the
AutoDL public ingress; Phase 2 remains running and immediately available as the
rollback service.
