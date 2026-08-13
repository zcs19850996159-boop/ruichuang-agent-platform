import assert from "node:assert/strict";
import {chromium, request} from "playwright";

const BATTERY_QUESTION = "如何按照手册更换电池？";
const CONTEXT_PRIME_QUESTION = "冰箱内照明灯不亮，如何按照手册更换灯泡？";
const EXPECTED_IMAGES = ["Manual27_1", "Manual27_2", "Manual27_3"];

function makeId(prefix) {
  return `${prefix}${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function normalizeBaseUrl(value) {
  return String(value || "").replace(/\/+$/, "");
}

function countPicMarkers(answer) {
  return (String(answer || "").match(/<PIC>/g) || []).length;
}

function imageIds(data) {
  return (Array.isArray(data?.images) ? data.images : [])
    .map((item) => typeof item === "string" ? item : item?.image_id)
    .filter(Boolean);
}

function manualId(data) {
  const candidates = [
    data?.route?.manual_id,
    data?.core_result?.selector?.route?.manual_id,
    data?.sources?.[0]?.manual_id,
    data?.evidence?.[0]?.manual_id,
  ];
  return String(candidates.find(Boolean) || "");
}

function parseSseBlock(block) {
  if (!block.trim()) return null;
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  return {event, payload: JSON.parse(dataLines.join("\n"))};
}

async function readSse(response) {
  assert.equal(response.ok(), true, `SSE request failed with HTTP ${response.status()}`);
  assert.match(
    response.headers()["content-type"] || "",
    /^text\/event-stream/,
    "Tool stream did not return text/event-stream",
  );
  const events = [];
  const body = await response.text();
  for (const block of body.split("\n\n")) {
    const parsed = parseSseBlock(block);
    if (parsed) events.push(parsed);
  }
  return events;
}

async function callToolStream(api, token) {
  const sessionId = makeId("e2e_stream_");
  const response = await api.post("/tools/v1/answer_customer_question", {
    headers: {
      Accept: "text/event-stream",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-Client-Type": "phase3-e2e-gate",
      "X-Knowledge-Space-Id": "competition",
      "X-Request-Id": makeId("e2e_req_"),
    },
    data: {
      schema_version: "1.0",
      question: BATTERY_QUESTION,
      attachments: [],
      conversation_context: {session_id: sessionId},
      response_mode: "stream",
    },
  });
  const events = await readSse(response);
  const errors = events.filter((item) => item.event === "error");
  assert.deepEqual(errors, [], "Tool stream emitted an error event");
  const deltas = events
    .filter((item) => item.event === "answer_delta")
    .map((item) => String(item.payload.text || ""))
    .join("");
  const finalEvents = events.filter((item) => item.event === "final");
  assert.equal(finalEvents.length, 1, "Tool stream must emit exactly one final event");
  const final = finalEvents[0].payload;
  assert.equal(deltas, String(final.data?.answer || ""), "Stream deltas differ from final answer");
  assert.deepEqual(imageIds(final.data), EXPECTED_IMAGES, "Stream returned the wrong image order");
  assert.equal(manualId(final.data), "Manual27", "Stream returned the wrong manual");
  assert.equal(countPicMarkers(final.data?.answer), EXPECTED_IMAGES.length);
  assert.equal(final.data?.escalation_required, false);
  return {
    event_counts: Object.fromEntries(
      [...new Set(events.map((item) => item.event))]
        .map((event) => [event, events.filter((item) => item.event === event).length]),
    ),
    manual_id: manualId(final.data),
    images: imageIds(final.data),
    streaming: final.data?.streaming || {},
  };
}

async function callSyncCompatibility(api, token) {
  const commonHeaders = {
    Accept: "application/json",
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    "X-Client-Type": "phase3-e2e-gate",
    "X-Request-Id": makeId("e2e_req_"),
  };
  const toolResponse = await api.post("/tools/v1/answer_customer_question", {
    headers: {
      ...commonHeaders,
      "X-Knowledge-Space-Id": "competition",
    },
    data: {
      schema_version: "1.0",
      question: BATTERY_QUESTION,
      attachments: [],
      conversation_context: {session_id: makeId("e2e_sync_")},
      response_mode: "sync",
    },
  });
  assert.equal(toolResponse.ok(), true, `Sync Tool API failed with HTTP ${toolResponse.status()}`);
  const toolPayload = await toolResponse.json();
  assert.deepEqual(imageIds(toolPayload.data), EXPECTED_IMAGES);
  assert.equal(manualId(toolPayload.data), "Manual27");
  assert.equal(countPicMarkers(toolPayload.data?.answer), EXPECTED_IMAGES.length);
  assert.equal(toolPayload.data?.escalation_required, false);

  const chatResponse = await api.post("/chat", {
    headers: commonHeaders,
    data: {
      question: BATTERY_QUESTION,
      images: [],
      session_id: makeId("e2e_chat_"),
      stream: false,
    },
  });
  assert.equal(chatResponse.ok(), true, `Official /chat failed with HTTP ${chatResponse.status()}`);
  const chatPayload = await chatResponse.json();
  assert.equal(chatPayload.code, 0);
  assert.deepEqual(imageIds(chatPayload.data), EXPECTED_IMAGES);
  assert.equal(manualId(chatPayload.data), "Manual27");
  assert.equal(countPicMarkers(chatPayload.data?.answer), EXPECTED_IMAGES.length);
  assert.equal(chatPayload.data?.answer_check?.constraint_pass, true);
  return {
    tool_images: imageIds(toolPayload.data),
    chat_images: imageIds(chatPayload.data),
    tool_manual_id: manualId(toolPayload.data),
    chat_manual_id: manualId(chatPayload.data),
    pic_count: countPicMarkers(chatPayload.data?.answer),
  };
}

async function latestAssistantState(page) {
  const messages = page.locator("#messages article.message.assistant");
  const count = await messages.count();
  if (!count) {
    return {
      count: 0,
      text: "",
      images: [],
      sendDisabled: false,
      status: "",
      step1: false,
      step2: false,
      step3: false,
    };
  }
  const last = messages.nth(count - 1);
  const state = await last.evaluate((node) => ({
    text: node.querySelector(".bubble")?.textContent || "",
    images: Array.from(node.querySelectorAll("figure img")).map((img) => img.alt || ""),
  }));
  state.count = count;
  state.sendDisabled = await page.locator("#send-button").isDisabled();
  state.status = (await page.locator("#composer-status").textContent()) || "";
  state.step1 = state.text.includes("按下鼠标底部的按钮");
  state.step2 = state.text.includes("打开电池仓");
  state.step3 = state.text.includes("装回电池仓盖");
  return state;
}

async function submitAndObserve(page, question, {recordSequence = false} = {}) {
  const priorCount = await page.locator("#messages article.message.assistant").count();
  await page.locator("#question-input").fill(question);
  const started = Date.now();
  await page.locator("#send-button").click();
  await page.waitForFunction(
    (count) => document.querySelectorAll("#messages article.message.assistant").length > count,
    priorCount,
  );

  const transitions = [];
  let lastKey = "";
  let finalState = null;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const current = await latestAssistantState(page);
    const done = current.status.includes("回答完成") && !current.sendDisabled;
    const key = JSON.stringify({
      images: current.images,
      done,
      busy: current.sendDisabled,
      step1: current.step1,
      step2: current.step2,
      step3: current.step3,
    });
    if (recordSequence && key !== lastKey) {
      transitions.push({
        elapsed_ms: Date.now() - started,
        images: current.images,
        answer_complete: done,
        response_in_progress: current.sendDisabled,
        step_1_visible: current.step1,
        step_2_visible: current.step2,
        step_3_visible: current.step3,
      });
      lastKey = key;
    }
    if (done) {
      finalState = current;
      break;
    }
    await page.waitForTimeout(25);
  }
  assert.ok(finalState, `Question did not complete within 30 seconds: ${question}`);
  return {
    elapsed_ms: Date.now() - started,
    final_state: finalState,
    transitions,
  };
}

async function runBrowserGate(baseUrl, token, headless) {
  const browser = await chromium.launch({headless});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  try {
    await page.goto(`${baseUrl}/workbench?e2e=${Date.now()}`, {waitUntil: "domcontentloaded"});
    await page.getByLabel("比赛 API Token", {exact: true}).fill(token);
    await page.getByRole("button", {name: "安全连接", exact: true}).click();
    await page.getByRole("heading", {name: "客服对话", exact: true}).waitFor();

    const originalSession = (await page.locator("#session-label").textContent()) || "";
    const contextPrime = await submitAndObserve(page, CONTEXT_PRIME_QUESTION);
    assert.match(contextPrime.final_state.text, /灯|lamp/i);

    const battery = await submitAndObserve(
      page,
      BATTERY_QUESTION,
      {recordSequence: true},
    );
    const firstInlineImage = battery.transitions.find(
      (item) => item.images.length > 0 && item.response_in_progress,
    );
    assert.ok(firstInlineImage, "No manual image appeared before answer completion");
    assert.deepEqual(
      firstInlineImage.images,
      ["Manual27_1"],
      "Later images appeared before the first inline image received a visible frame",
    );
    assert.equal(firstInlineImage.step_1_visible, true);
    assert.equal(firstInlineImage.step_3_visible, false);
    assert.equal(firstInlineImage.response_in_progress, true);

    const twoImages = battery.transitions.find((item) => item.images.length === 2);
    assert.ok(twoImages, "Second manual image was not observed during streaming");
    assert.deepEqual(twoImages.images, ["Manual27_1", "Manual27_2"]);
    assert.equal(twoImages.answer_complete, false);

    const threeImagesBeforeFinal = battery.transitions.find(
      (item) => item.images.length === 3 && item.response_in_progress,
    );
    assert.ok(threeImagesBeforeFinal, "All three images appeared only after final completion");
    assert.deepEqual(threeImagesBeforeFinal.images, EXPECTED_IMAGES);
    assert.deepEqual(battery.final_state.images, EXPECTED_IMAGES);
    assert.equal(battery.final_state.step1, true);
    assert.equal(battery.final_state.step2, true);
    assert.equal(battery.final_state.step3, true);
    assert.equal(battery.final_state.text.includes("人工客服"), false);
    assert.equal(await page.locator("#trace-version").textContent(), "competition-kb-v1");
    assert.equal(await page.locator("#trace-validation").textContent(), "通过");
    assert.match(await page.locator("#evidence-list").textContent(), /Manual27/);

    await page.locator("#new-chat").click();
    await page.waitForFunction(
      (previous) => document.querySelector("#session-label")?.textContent !== previous,
      originalSession,
    );
    const resetSession = (await page.locator("#session-label").textContent()) || "";
    assert.notEqual(resetSession, originalSession);
    assert.equal(await page.locator("#messages article.message").count(), 0);
    assert.equal(await page.locator("#welcome").isVisible(), true);

    return {
      context_prime_question: CONTEXT_PRIME_QUESTION,
      explicit_battery_question_after_context_prime: BATTERY_QUESTION,
      context_override_protection: true,
      original_session: originalSession.replace(/^会话\s*/, ""),
      reset_session: resetSession.replace(/^会话\s*/, ""),
      session_changed: true,
      first_inline_image_state: firstInlineImage,
      three_images_before_final_state: threeImagesBeforeFinal,
      final_elapsed_ms: battery.elapsed_ms,
      final_images: battery.final_state.images,
      trace_version: "competition-kb-v1",
      validation: "passed",
      escalation_required: false,
      transitions: battery.transitions,
    };
  } finally {
    await context.close();
    await browser.close();
  }
}

async function checkRollback(url) {
  if (!url) return {configured: false, passed: false};
  const rollbackApi = await request.newContext({baseURL: url});
  try {
    for (const path of ["/ready", "/health"]) {
      const response = await rollbackApi.get(path).catch(() => null);
      if (response?.ok()) {
        return {
          configured: true,
          passed: true,
          path,
          status: response.status(),
        };
      }
    }
  } finally {
    await rollbackApi.dispose();
  }
  assert.fail(`Rollback service is not healthy at ${url}`);
}

export async function runWorkbenchStreamingGate({
  baseUrl,
  token,
  rollbackBaseUrl = "",
  headless = true,
} = {}) {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const normalizedRollbackUrl = normalizeBaseUrl(rollbackBaseUrl);
  assert.ok(normalizedBaseUrl, "baseUrl is required");
  assert.ok(String(token || "").trim(), "token is required");

  const started = Date.now();
  const api = await request.newContext({baseURL: normalizedBaseUrl});
  let streamContract;
  let compatibility;
  try {
    const readiness = await api.get("/ready");
    assert.equal(
      readiness.ok(),
      true,
      `Phase 3 readiness failed with HTTP ${readiness.status()}`,
    );
    streamContract = await callToolStream(api, token);
    compatibility = await callSyncCompatibility(api, token);
  } finally {
    await api.dispose();
  }
  const browserGate = await runBrowserGate(normalizedBaseUrl, token, headless);
  const rollback = await checkRollback(normalizedRollbackUrl);

  return {
    schema_version: "1.0",
    gate: "phase3-workbench-streaming-e2e",
    status: "passed",
    executed_at: new Date().toISOString(),
    elapsed_ms: Date.now() - started,
    base_url: normalizedBaseUrl,
    checks: {
      readiness: true,
      tool_stream_contract: streamContract,
      sync_and_official_compatibility: compatibility,
      browser: browserGate,
      rollback,
    },
  };
}
