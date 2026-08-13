import assert from "node:assert/strict";

const BATTERY_QUESTION = "如何按照手册更换电池？";
const CONTEXT_PRIME_QUESTION = "冰箱内照明灯不亮，如何按照手册更换灯泡？";
const EXPECTED_IMAGES = ["Manual27_1", "Manual27_2", "Manual27_3"];

function assistantBlock(snapshot) {
  const marker = "- article:\n    - generic: R\n    - strong: 睿创客服";
  const start = snapshot.lastIndexOf(marker);
  if (start < 0) return "";
  const tail = snapshot.slice(start);
  const end = tail.indexOf('\n  - generic "添加图片"');
  return end < 0 ? tail : tail.slice(0, end);
}

function sessionId(snapshot) {
  const match = snapshot.match(/generic: 会话 ([^\s]+)/);
  return match?.[1] || "";
}

function snapshotState(snapshot) {
  const answer = assistantBlock(snapshot);
  return {
    images: [...new Set(answer.match(/Manual27_[123]/g) || [])],
    answer_complete: snapshot.includes("generic: 回答完成"),
    response_in_progress: snapshot.includes('button "发送问题" [disabled]'),
    step_1_visible: answer.includes("按下鼠标底部的按钮"),
    step_2_visible: answer.includes("打开电池仓"),
    step_3_visible: answer.includes("装回电池仓盖"),
    answer,
  };
}

async function submitAndObserve(tab, question, {recordSequence = false} = {}) {
  const playwright = tab.playwright;
  await playwright
    .getByRole("textbox", {
      name: "输入产品操作、故障或售后问题…",
      exact: true,
    })
    .fill(question);
  const started = Date.now();
  await playwright.getByRole("button", {name: "发送问题", exact: true}).click();

  const transitions = [];
  let lastKey = "";
  let finalState = null;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const snapshot = await playwright.domSnapshot();
    const current = snapshotState(snapshot);
    const key = JSON.stringify({
      images: current.images,
      done: current.answer_complete,
      busy: current.response_in_progress,
      step1: current.step_1_visible,
      step2: current.step_2_visible,
      step3: current.step_3_visible,
    });
    if (recordSequence && key !== lastKey) {
      transitions.push({
        elapsed_ms: Date.now() - started,
        images: current.images,
        answer_complete: current.answer_complete,
        response_in_progress: current.response_in_progress,
        step_1_visible: current.step_1_visible,
        step_2_visible: current.step_2_visible,
        step_3_visible: current.step_3_visible,
      });
      lastKey = key;
    }
    if (current.answer_complete && !current.response_in_progress) {
      finalState = current;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.ok(finalState, `Question did not complete within 30 seconds: ${question}`);
  return {
    elapsed_ms: Date.now() - started,
    final_state: finalState,
    transitions,
  };
}

export async function runWorkbenchBrowserClientGate({
  tab,
  baseUrl,
  token,
} = {}) {
  assert.ok(tab, "tab is required");
  assert.ok(String(baseUrl || "").trim(), "baseUrl is required");
  assert.ok(String(token || "").trim(), "token is required");

  const started = Date.now();
  await tab.goto(`${String(baseUrl).replace(/\/+$/, "")}/workbench?e2e=${started}`);
  const playwright = tab.playwright;
  let snapshot = await playwright.domSnapshot();
  assert.match(snapshot, /进入聊天工作台/);
  await playwright
    .getByLabel("比赛 API Token", {exact: true})
    .fill(token);
  await playwright
    .getByRole("button", {name: "安全连接", exact: true})
    .click({timeout: 10000})
    .catch(() => {});

  const loginDeadline = Date.now() + 10000;
  while (Date.now() < loginDeadline) {
    snapshot = await playwright.domSnapshot();
    if (snapshot.includes('heading "客服对话"')) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.match(snapshot, /heading "客服对话"/);
  const originalSession = sessionId(snapshot);
  assert.ok(originalSession, "Initial workbench session ID is missing");

  const contextPrime = await submitAndObserve(tab, CONTEXT_PRIME_QUESTION);
  assert.match(contextPrime.final_state.answer, /灯|lamp/i);

  const battery = await submitAndObserve(
    tab,
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
  assert.equal(battery.final_state.step_1_visible, true);
  assert.equal(battery.final_state.step_2_visible, true);
  assert.equal(battery.final_state.step_3_visible, true);
  assert.equal(battery.final_state.answer.includes("人工客服"), false);

  snapshot = await playwright.domSnapshot();
  assert.match(snapshot, /strong: competition-kb-v1/);
  assert.match(snapshot, /generic: 校验\n    - strong: 通过/);
  assert.match(snapshot, /strong: Manual27/);

  await playwright
    .getByRole("button", {name: "＋ 新建会话", exact: true})
    .click();
  const resetDeadline = Date.now() + 5000;
  let resetSession = "";
  while (Date.now() < resetDeadline) {
    snapshot = await playwright.domSnapshot();
    resetSession = sessionId(snapshot);
    if (resetSession && resetSession !== originalSession) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.ok(resetSession, "Reset workbench session ID is missing");
  assert.notEqual(resetSession, originalSession);
  assert.match(snapshot, /今天需要解决什么问题/);
  assert.equal(snapshot.includes("- article:"), false);

  return {
    schema_version: "1.0",
    gate: "phase3-workbench-browser-client-e2e",
    status: "passed",
    executed_at: new Date().toISOString(),
    elapsed_ms: Date.now() - started,
    checks: {
      context_prime_question: CONTEXT_PRIME_QUESTION,
      explicit_battery_question_after_context_prime: BATTERY_QUESTION,
      context_override_protection: true,
      first_inline_image_state: firstInlineImage,
      three_images_before_final_state: threeImagesBeforeFinal,
      final_elapsed_ms: battery.elapsed_ms,
      final_images: battery.final_state.images,
      trace_version: "competition-kb-v1",
      validation: "passed",
      escalation_required: false,
      original_session: originalSession,
      reset_session: resetSession,
      session_changed: true,
      transitions: battery.transitions,
    },
  };
}
