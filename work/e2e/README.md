# Phase 3 workbench end-to-end gate

This gate exercises the deployed application rather than a mocked DOM.

It verifies:

- Phase 3 readiness;
- Tool API SSE deltas equal the final answer;
- sync Tool API and official `/chat` compatibility;
- an unrelated refrigerator conversation cannot override a later explicit
  Manual27 battery question in the same session;
- `Manual27_1` appears after step 1 while the response is still running and
  before step 3 exists;
- all three figures appear in `<PIC>` order before final completion;
- final trace, validation, evidence, and no-escalation behavior;
- the New Chat control changes the session and clears prior messages;
- optional Phase 2 rollback health.

Install the pinned browser dependency in this directory:

```bash
npm install
```

Run with credentials supplied only through the environment:

```bash
CUSTOMER_SERVICE_API_TOKEN='...' \
WORKBENCH_BASE_URL='http://127.0.0.1:18878' \
ROLLBACK_BASE_URL='http://127.0.0.1:18879' \
npm run test:streaming
```

The token is never written to the report. `ROLLBACK_BASE_URL` is optional, but
release acceptance should provide it so that rollback readiness is a hard gate.

In a Codex managed environment, localhost access for a standalone Playwright
process may be intentionally blocked. In that environment,
`workbench_browser_client_gate.mjs` exports the same user-visible browser gate
for execution against an authorized Codex Browser tab. The standard Playwright
gate remains the CI entry point.

The server-side API and rollback half of the gate can be run independently:

```bash
PYTHONPATH=work \
CUSTOMER_SERVICE_API_TOKEN='...' \
WORKBENCH_BASE_URL='http://127.0.0.1:8878' \
ROLLBACK_BASE_URL='http://127.0.0.1:8877' \
.venv/bin/python work/e2e/api_compatibility_gate.py
```

## Legacy official TXT ingestion gate

`legacy_txt_ingestion_gate.py` verifies the full administrative lifecycle for
the official `[manual text, image id list]` TXT format. It creates an isolated
enterprise tenant and knowledge space, publishes one old manual as v1, appends
a second old manual into an aggregate v2, checks managed retrieval and image
delivery, rolls back to v1, and restores v2. When `KAFU_API_TOKEN` is present,
it also verifies that the frozen Competition Profile remains unchanged.

Required server-side environment:

```bash
export CONTROL_PLANE_BOOTSTRAP_TOKEN='...'
export KNOWLEDGE_REFERENCE_IMAGE_ROOTS='/approved/manual/image/root'
export TXT_BASELINE_SOURCE='/approved/source/空气净化器手册.txt'
export TXT_APPENDED_SOURCE='/approved/source/人体工学椅手册.txt'
PYTHONPATH=work .venv/bin/python work/e2e/legacy_txt_ingestion_gate.py
```

The script never prints the bootstrap, enterprise, or competition credentials.
