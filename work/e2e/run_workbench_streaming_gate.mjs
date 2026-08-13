import {writeFile} from "node:fs/promises";
import {resolve} from "node:path";

import {runWorkbenchStreamingGate} from "./workbench_streaming_gate.mjs";

const token = String(process.env.CUSTOMER_SERVICE_API_TOKEN || "").trim();
if (!token) {
  throw new Error("CUSTOMER_SERVICE_API_TOKEN is required");
}

const report = await runWorkbenchStreamingGate({
  baseUrl: process.env.WORKBENCH_BASE_URL || "http://127.0.0.1:18878",
  rollbackBaseUrl: process.env.ROLLBACK_BASE_URL || "",
  token,
  headless: process.env.E2E_HEADLESS !== "0",
});

const output = resolve(
  process.env.E2E_REPORT_PATH
    || "deliverables/phase3/phase3_workbench_streaming_e2e_latest.json",
);
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, {mode: 0o600});
process.stdout.write(`${JSON.stringify({status: report.status, report: output})}\n`);
