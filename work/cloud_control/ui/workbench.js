"use strict";

const byId = (id) => document.getElementById(id);
const state = {
  mode: "competition",
  token: "",
  tenant: "",
  identity: null,
  space: "",
  sessionId: "",
  attachments: [],
  objectUrls: [],
  imageBlobCache: new Map(),
  imageBlobPromises: new Map(),
  conversations: [],
  currentConversationId: "",
  historyKey: "",
  busy: false,
};

const HISTORY_VERSION = 1;
const HISTORY_MAX_CONVERSATIONS = 20;
const HISTORY_MAX_MESSAGES = 20;
const HISTORY_MAX_TEXT = 8000;
const HISTORY_RETENTION_MS = 90 * 24 * 60 * 60 * 1000;

function makeId(prefix) {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  return `${prefix}${Date.now().toString(36)}-${Array.from(values, (x) => x.toString(36)).join("-")}`;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function authHeaders(extra = {}) {
  return {
    Accept: "application/json",
    Authorization: `Bearer ${state.token}`,
    ...extra,
  };
}

async function api(path, options = {}) {
  const headers = authHeaders(options.headers || {});
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {...options, headers});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.error) {
    const message = payload.error?.message || payload.error?.error_code || payload.detail || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload.data;
}

async function toolApi(path, options = {}) {
  const headers = authHeaders(options.headers || {});
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {...options, headers});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.error) {
    const message = payload.error?.message || payload.error?.error_code || payload.detail || `HTTP ${response.status}`;
    throw new Error(message);
  }
  if (!payload.data) throw new Error("客服核心没有返回有效数据");
  return payload;
}

async function streamToolApi(path, options = {}, handlers = {}) {
  const headers = authHeaders({
    Accept: "text/event-stream",
    ...(options.headers || {}),
  });
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload.error?.message || payload.detail || `HTTP ${response.status}`;
    throw new Error(message);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult = null;

  const consumeBlock = async (block) => {
    if (!block.trim()) return;
    let event = "message";
    const dataLines = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return;
    const payload = JSON.parse(dataLines.join("\n"));
    if (event === "error") {
      throw new Error(payload.message || payload.error_code || "流式请求失败");
    }
    if (event === "final") {
      finalResult = payload;
      return;
    }
    await handlers[event]?.(payload);
  };

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) await consumeBlock(block);
    if (done) break;
  }
  if (buffer.trim()) await consumeBlock(buffer);
  if (!finalResult?.data) throw new Error("流式连接结束但没有收到最终答案");
  return finalResult;
}

function cleanupObjectUrls() {
  for (const url of state.objectUrls) URL.revokeObjectURL(url);
  state.objectUrls = [];
  state.imageBlobCache.clear();
  state.imageBlobPromises.clear();
}

function conversationStorageKey() {
  if (!state.identity || !state.space) return "";
  const scope = [
    state.mode,
    state.tenant || state.identity.tenant_id || "default",
    state.identity.user_id || "anonymous",
    state.space,
  ].map((item) => encodeURIComponent(String(item)));
  return `ruichuang-workbench-history:v${HISTORY_VERSION}:${scope.join(":")}`;
}

function conversationTitle(question) {
  const compact = String(question || "").replace(/\s+/g, " ").trim();
  if (!compact) return "新会话";
  return compact.length > 28 ? `${compact.slice(0, 28)}…` : compact;
}

function safeHistoryText(text) {
  return String(text || "").slice(0, HISTORY_MAX_TEXT);
}

function safeManualImageUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    const parsed = new URL(raw, window.location.origin);
    if (parsed.origin !== window.location.origin) return "";
    if (!parsed.pathname.startsWith("/manual-images/")) return "";
    return `${parsed.pathname}${parsed.search}`;
  } catch (_) {
    return "";
  }
}

function storageGet(key) {
  if (!key) return "";
  try {
    return localStorage.getItem(key) || "";
  } catch (_) {
    return "";
  }
}

function storageSet(key, value) {
  if (!key) return false;
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (_) {
    return false;
  }
}

function storageRemove(key) {
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch (_) {
    // The workbench still functions when private browsing blocks local storage.
  }
}

function activeConversationStorageKey() {
  return state.historyKey ? `${state.historyKey}:active` : "";
}

function safeImageRefs(data) {
  return imageReferences(data).slice(0, 8).map((item) => ({
    image_id: String(item.image_id || "").slice(0, 128),
    url: safeManualImageUrl(item.url).slice(0, 512),
  }));
}

function safeTraceData(data) {
  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  return {
    knowledge_version: String(data.knowledge_version || "").slice(0, 128),
    confidence: data.confidence ?? data.core_result?.selector?.confidence?.score ?? null,
    escalation_required: Boolean(data.escalation_required),
    validation: data.validation && typeof data.validation === "object"
      ? {pass: data.validation.pass, ok: data.validation.ok}
      : {},
    evidence: evidence.slice(0, 8).map((item) => ({
      title: safeHistoryText(item.title || ""),
      document_id: safeHistoryText(item.document_id || ""),
      manual_id: safeHistoryText(item.manual_id || ""),
      product: safeHistoryText(item.product || ""),
      source_ref: safeHistoryText(item.source_ref || ""),
      section: safeHistoryText(item.section || ""),
      page: item.page ?? null,
      chunk_id: safeHistoryText(item.chunk_id || ""),
      score: item.score ?? null,
      evidence_snippet: safeHistoryText(
        item.evidence_snippet || item.text || item.content || "",
      ).slice(0, 600),
    })),
  };
}

function validConversation(item) {
  return item && typeof item === "object"
    && typeof item.sessionId === "string"
    && item.sessionId.length > 0
    && Array.isArray(item.messages);
}

function saveConversationHistory() {
  if (!state.historyKey) return;
  const now = Date.now();
  state.conversations = state.conversations
    .filter(validConversation)
    .filter((item) => now - Number(item.updatedAt || item.createdAt || now) <= HISTORY_RETENTION_MS)
    .sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0))
    .slice(0, HISTORY_MAX_CONVERSATIONS)
    .map((item) => ({...item, messages: item.messages.slice(-HISTORY_MAX_MESSAGES)}));
  const payload = JSON.stringify({version: HISTORY_VERSION, conversations: state.conversations});
  try {
    localStorage.setItem(state.historyKey, payload);
  } catch (error) {
    state.conversations = state.conversations.slice(0, 10).map((item) => ({
      ...item,
      messages: item.messages.slice(-12),
    }));
    try {
      localStorage.setItem(
        state.historyKey,
        JSON.stringify({version: HISTORY_VERSION, conversations: state.conversations}),
      );
    } catch (_) {
      byId("composer-status").textContent = "浏览器存储空间不足，最近聊天未能保存";
    }
  }
  renderRecentChats();
}

function loadConversationHistory() {
  state.historyKey = conversationStorageKey();
  state.conversations = [];
  state.currentConversationId = "";
  if (!state.historyKey) {
    renderRecentChats();
    return;
  }
  try {
    const parsed = JSON.parse(storageGet(state.historyKey) || "{}");
    state.conversations = Array.isArray(parsed.conversations)
      ? parsed.conversations.filter(validConversation)
      : [];
  } catch (_) {
    state.conversations = [];
  }
  saveConversationHistory();
}

function recentTime(timestamp) {
  const value = Number(timestamp || 0);
  if (!value) return "";
  const elapsed = Math.max(0, Date.now() - value);
  if (elapsed < 60_000) return "刚刚";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)} 分钟前`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)} 小时前`;
  if (elapsed < 7 * 86_400_000) return `${Math.floor(elapsed / 86_400_000)} 天前`;
  return new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric"}).format(value);
}

function renderRecentChats() {
  const list = byId("recent-chats");
  if (!list) return;
  list.replaceChildren();
  if (!state.conversations.length) {
    list.append(element("div", "发送第一条消息后，会话会显示在这里", "recent-empty"));
    return;
  }
  for (const conversation of state.conversations) {
    const row = element("div", undefined, "recent-chat-row");
    row.classList.toggle("active", conversation.sessionId === state.currentConversationId);
    const open = element("button", undefined, "recent-chat-open");
    open.type = "button";
    open.title = conversation.title || "历史会话";
    open.append(
      element("strong", conversation.title || "历史会话"),
      element("span", recentTime(conversation.updatedAt)),
    );
    open.addEventListener("click", () => openConversation(conversation.sessionId));
    const remove = element("button", "×", "recent-chat-delete");
    remove.type = "button";
    remove.title = "删除这条最近聊天";
    remove.setAttribute("aria-label", `删除 ${conversation.title || "历史会话"}`);
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.sessionId);
    });
    row.append(open, remove);
    list.append(row);
  }
}

function ensureCurrentConversation(question) {
  let conversation = state.conversations.find(
    (item) => item.sessionId === state.currentConversationId,
  );
  if (!conversation) {
    const now = Date.now();
    conversation = {
      sessionId: state.sessionId,
      title: conversationTitle(question),
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    state.currentConversationId = conversation.sessionId;
    state.conversations.unshift(conversation);
    storageSet(activeConversationStorageKey(), state.currentConversationId);
  }
  return conversation;
}

function recordHistoryMessage(message) {
  const conversation = state.conversations.find(
    (item) => item.sessionId === state.currentConversationId,
  );
  if (!conversation) return;
  conversation.messages.push(message);
  conversation.messages = conversation.messages.slice(-HISTORY_MAX_MESSAGES);
  conversation.updatedAt = Date.now();
  saveConversationHistory();
}

function openConversation(sessionId) {
  if (state.busy) return;
  const conversation = state.conversations.find((item) => item.sessionId === sessionId);
  if (!conversation) return;
  cleanupObjectUrls();
  state.sessionId = conversation.sessionId;
  state.currentConversationId = conversation.sessionId;
  state.attachments = [];
  byId("attachment-preview").replaceChildren();
  byId("question-input").value = "";
  byId("messages").replaceChildren();
  let lastAssistant = null;
  for (const message of conversation.messages) {
    if (message.role === "user") {
      appendMessage("user", (bubble) => {
        bubble.append(element("p", message.text || ""));
        if (message.attachmentCount) {
          bubble.append(element(
            "p",
            `包含 ${message.attachmentCount} 张用户图片（历史记录不缓存上传原图）`,
            "message-meta",
          ));
        }
      });
    } else if (message.role === "assistant") {
      appendMessage("assistant", (bubble) => {
        appendAnswerParts(
          bubble,
          message.text || "",
          safeImageRefs({image_refs: message.imageRefs || []}),
        );
      }, message.metaText || "历史回答");
      lastAssistant = message;
    }
  }
  if (!conversation.messages.length) byId("messages").append(buildWelcome());
  if (lastAssistant?.traceData) {
    renderTrace(lastAssistant.traceData, lastAssistant.elapsedMs);
  } else {
    byId("trace-empty").hidden = false;
    byId("trace-content").hidden = true;
  }
  byId("session-label").textContent = `会话 ${state.sessionId}`;
  byId("composer-status").textContent = "已恢复最近聊天，可以继续追问";
  storageSet(activeConversationStorageKey(), state.sessionId);
  renderRecentChats();
  scrollMessages();
}

function deleteConversation(sessionId) {
  if (state.busy) return;
  state.conversations = state.conversations.filter((item) => item.sessionId !== sessionId);
  if (state.currentConversationId === sessionId) resetConversation();
  saveConversationHistory();
  fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  }).catch(() => {});
}

function resetConversation() {
  cleanupObjectUrls();
  state.sessionId = makeId("wb_session_");
  state.currentConversationId = "";
  state.attachments = [];
  byId("messages").replaceChildren();
  byId("messages").append(buildWelcome());
  byId("attachment-preview").replaceChildren();
  byId("question-input").value = "";
  byId("trace-empty").hidden = false;
  byId("trace-content").hidden = true;
  byId("session-label").textContent = `会话 ${state.sessionId}`;
  storageRemove(activeConversationStorageKey());
  renderRecentChats();
  byId("composer-status").textContent = state.mode === "competition"
    ? "比赛 Profile · 一次回答仅调用客服核心一次"
    : "企业 Profile · 一次回答仅调用客服核心一次";
}

function buildWelcome() {
  const wrap = element("div", undefined, "welcome");
  wrap.id = "welcome";
  wrap.append(element("div", "R", "welcome-orb"));
  wrap.append(element("p", "RUICHUANG AI", "eyebrow"));
  wrap.append(element("h2", "今天需要解决什么问题？"));
  wrap.append(element(
    "p",
    state.mode === "competition"
      ? "我会调用冻结的比赛知识库，按官方回答协议返回文本、图片与证据。"
      : "我会调用已发布的企业知识，返回带来源、置信度和图片证据的回答。",
  ));
  return wrap;
}

function scrollMessages() {
  const messages = byId("messages");
  requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
}

function appendMessage(role, bodyBuilder, metaText = "") {
  byId("welcome")?.remove();
  const wrapper = element("article", undefined, `message ${role}`);
  const head = element("div", undefined, "message-head");
  head.append(element("span", role === "user" ? "你" : "R"));
  head.append(element("strong", role === "user" ? "你" : "睿创客服"));
  wrapper.append(head);
  const bubble = element("div", undefined, "bubble");
  bodyBuilder(bubble);
  wrapper.append(bubble);
  if (metaText) wrapper.append(element("div", metaText, "message-meta"));
  byId("messages").append(wrapper);
  scrollMessages();
  return {wrapper, bubble};
}

function appendUserMessage(question, attachments) {
  appendMessage("user", (bubble) => {
    bubble.append(element("p", question));
    if (attachments.length) {
      const images = element("div", undefined, "message-images");
      for (const attachment of attachments) {
        const img = document.createElement("img");
        img.src = attachment.previewUrl;
        img.alt = attachment.name;
        images.append(img);
      }
      bubble.append(images);
    }
  });
}

function appendThinking() {
  return appendMessage("assistant", (bubble) => {
    const thinking = element("span", undefined, "thinking");
    thinking.append(element("i"), element("i"), element("i"));
    bubble.append(thinking);
  });
}

function createStreamingAnswer(pending) {
  return {
    pending,
    answer: "",
    imageRefs: [],
    figures: new Map(),
  };
}

function visibleStreamingAnswer(answer) {
  return String(answer || "").replace(/<(?:P(?:I(?:C)?)?)?$/, "");
}

function streamingFigure(stream, index) {
  let entry = stream.figures.get(index);
  if (!entry) {
    entry = {
      node: element("figure", undefined, "answer-figure"),
      imageKey: "",
      loadPromise: null,
    };
    stream.figures.set(index, entry);
  }

  const imageRef = stream.imageRefs[index];
  const imageKey = imageRef?.url
    ? `${imageRef.image_id || ""}|${imageRef.url}`
    : "";
  if (imageKey && entry.imageKey !== imageKey) {
    entry.imageKey = imageKey;
    const loading = element("span", "正在加载对应手册图片…", "muted");
    entry.node.replaceChildren(loading);
    entry.loadPromise = loadEvidenceImage(imageRef, entry.node, loading);
  } else if (!imageKey && !entry.imageKey) {
    entry.node.replaceChildren(
      element("span", "正在定位对应手册图片…", "muted"),
    );
  }
  return entry.node;
}

function completePicCount(answer) {
  return (String(answer || "").match(/<PIC>/g) || []).length;
}

async function waitForInlineImagePaint(stream, previousPicCount) {
  const currentPicCount = completePicCount(stream.answer);
  if (currentPicCount <= previousPicCount) return;
  const imageLoads = [];
  for (let index = previousPicCount; index < currentPicCount; index += 1) {
    const pending = stream.figures.get(index)?.loadPromise;
    if (pending) imageLoads.push(pending);
  }
  if (imageLoads.length) {
    await Promise.race([
      Promise.allSettled(imageLoads),
      new Promise((resolve) => setTimeout(resolve, 600)),
    ]);
  }
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise((resolve) => setTimeout(resolve, 100));
}

function renderStreamingAnswer(stream, final = false) {
  const visible = final
    ? String(stream.answer || "")
    : visibleStreamingAnswer(stream.answer);
  const parts = visible.split("<PIC>");
  const nodes = [];
  for (let index = 0; index < parts.length; index += 1) {
    if (parts[index].trim()) {
      nodes.push(element("p", parts[index].trim(), final ? "" : "streaming-text"));
    }
    if (index < parts.length - 1) {
      nodes.push(streamingFigure(stream, index));
    }
  }
  if (!nodes.length) {
    nodes.push(element("p", "正在生成回答…", "streaming-text"));
  }
  if (!final) {
    const cursor = element("span", undefined, "stream-cursor");
    cursor.setAttribute("aria-hidden", "true");
    nodes.push(cursor);
  }
  stream.pending.bubble.replaceChildren(...nodes);
  scrollMessages();
}

function resetStreamingAnswer(stream) {
  stream.answer = "";
  stream.figures.clear();
  stream.pending.bubble.replaceChildren(
    element("p", "初稿未通过校验，正在重新生成…", "muted"),
  );
  scrollMessages();
}

function appendAnswerParts(container, answer, imageRefs) {
  const parts = String(answer || "").split("<PIC>");
  for (let index = 0; index < parts.length; index += 1) {
    if (parts[index].trim()) container.append(element("p", parts[index].trim()));
    if (index < parts.length - 1) {
      const imageRef = imageRefs[index];
      const figure = element("figure", undefined, "answer-figure");
      if (imageRef?.url) {
        const loading = element("span", "正在加载对应手册图片…", "muted");
        figure.append(loading);
        loadEvidenceImage(imageRef, figure, loading);
      } else {
        figure.append(element("span", "回答引用了图片，但当前知识版本没有返回可用图片。", "muted"));
      }
      container.append(figure);
    }
  }
}

async function loadEvidenceImage(imageRef, figure, loading) {
  try {
    const cacheKey = `${imageRef.image_id || ""}|${imageRef.url}`;
    let url = state.imageBlobCache.get(cacheKey);
    if (!url) {
      let pending = state.imageBlobPromises.get(cacheKey);
      if (!pending) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 12_000);
        pending = fetch(imageRef.url, {
          headers: authHeaders(),
          signal: controller.signal,
        }).then(async (response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const blobUrl = URL.createObjectURL(await response.blob());
          state.objectUrls.push(blobUrl);
          state.imageBlobCache.set(cacheKey, blobUrl);
          return blobUrl;
        }).finally(() => {
          clearTimeout(timeout);
          state.imageBlobPromises.delete(cacheKey);
        });
        state.imageBlobPromises.set(cacheKey, pending);
      }
      url = await pending;
    }
    if (!figure.isConnected) return;
    const img = document.createElement("img");
    img.src = url;
    img.alt = imageRef.image_id || "手册图片";
    figure.replaceChildren(img, element("figcaption", imageRef.image_id || "手册图片"));
  } catch (error) {
    loading.textContent = `图片加载失败：${error instanceof Error ? error.message : String(error)}`;
  }
}

function renderTrace(data, elapsedMs) {
  byId("trace-empty").hidden = true;
  byId("trace-content").hidden = false;
  byId("trace-profile").textContent = state.mode === "competition" ? "Competition" : "Enterprise";
  byId("trace-space").textContent = state.space || "—";
  byId("trace-version").textContent = data.knowledge_version || "未发布";
  const rawConfidence = data.confidence ?? data.core_result?.selector?.confidence?.score;
  const confidence = rawConfidence === null || rawConfidence === undefined || rawConfidence === ""
    ? Number.NaN
    : Number(rawConfidence);
  byId("trace-confidence").textContent = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "—";
  byId("trace-elapsed").textContent = Number.isFinite(Number(elapsedMs)) ? `${Math.round(Number(elapsedMs))} ms` : "—";
  const valid = typeof data.validation?.pass === "boolean"
    ? data.validation.pass
    : data.validation?.ok;
  byId("trace-validation").textContent = valid === true ? "通过" : valid === false ? "需复核" : "—";

  const escalation = byId("escalation-note");
  escalation.hidden = !data.escalation_required;
  escalation.textContent = data.escalation_required ? "当前证据不足或答案未通过校验，建议转人工客服继续处理。" : "";

  const evidence = Array.isArray(data.evidence) ? data.evidence : [];
  byId("evidence-count").textContent = String(evidence.length);
  const list = byId("evidence-list");
  list.replaceChildren();
  if (!evidence.length) {
    list.append(element("p", "当前回答没有可展示的已发布证据。", "muted"));
    return;
  }
  for (const item of evidence.slice(0, 8)) {
    const card = element("article", undefined, "evidence-card");
    card.append(element("strong", item.title || item.document_id || item.manual_id || item.product || item.source_ref || "知识库证据"));
    const details = [
      item.section,
      item.page ? `第 ${item.page} 页` : "",
      item.chunk_id || "",
      Number.isFinite(Number(item.score)) ? `相关度 ${Number(item.score).toFixed(3)}` : "",
    ].filter(Boolean);
    card.append(element("span", details.join(" · ")));
    const snippet = item.evidence_snippet || item.text || item.content;
    if (snippet) card.append(element("p", String(snippet).slice(0, 220)));
    list.append(card);
  }
}

function imageReferences(data) {
  const references = Array.isArray(data.image_refs) && data.image_refs.length
    ? data.image_refs
    : (Array.isArray(data.images) ? data.images : []);
  return references.map((item) => {
    const imageId = typeof item === "string" ? item : item?.image_id;
    const candidateUrl = typeof item === "object" && item?.url
      ? item.url
      : (imageId ? `/manual-images/${encodeURIComponent(imageId)}` : "");
    return {
      image_id: String(imageId || "").slice(0, 128),
      url: safeManualImageUrl(candidateUrl),
    };
  }).filter((item) => item.image_id && item.url);
}

function renderAnswer(result, pending, stream = null) {
  const data = result.data || {};
  const answer = String(data.answer || "没有返回答案。");
  if (stream) {
    for (const child of Array.from(pending.wrapper.children)) {
      if (child.classList.contains("message-meta")) child.remove();
    }
    stream.answer = answer;
    stream.imageRefs = imageReferences(data);
    renderStreamingAnswer(stream, true);
    if (data.escalation_required) {
      pending.bubble.append(element("p", "此问题需要人工客服继续处理。", "message-meta"));
    }
    pending.wrapper.append(element(
      "div",
      `知识版本 ${data.knowledge_version || "—"} · ${Math.round(Number(result.elapsed_ms) || 0)} ms`,
      "message-meta",
    ));
    renderTrace(data, result.elapsed_ms);
    return;
  }
  pending.wrapper.remove();
  appendMessage("assistant", (bubble) => {
    appendAnswerParts(bubble, answer, imageReferences(data));
    if (data.escalation_required) bubble.append(element("p", "此问题需要人工客服继续处理。", "message-meta"));
  }, `知识版本 ${data.knowledge_version || "—"} · ${Math.round(Number(result.elapsed_ms) || 0)} ms`);
  renderTrace(data, result.elapsed_ms);
}

function renderError(error, pending) {
  pending.bubble.classList.add("error-bubble");
  pending.bubble.replaceChildren(element("p", error instanceof Error ? error.message : String(error)));
}

function renderAttachmentPreview() {
  const preview = byId("attachment-preview");
  preview.replaceChildren();
  state.attachments.forEach((attachment, index) => {
    const item = element("div", undefined, "attachment-item");
    const img = document.createElement("img");
    img.src = attachment.previewUrl;
    img.alt = attachment.name;
    const remove = element("button", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `移除 ${attachment.name}`);
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(attachment.previewUrl);
      state.attachments.splice(index, 1);
      renderAttachmentPreview();
    });
    item.append(img, remove);
    preview.append(item);
  });
}

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("图片读取失败"));
    reader.readAsDataURL(file);
  });
}

async function acceptFiles(files) {
  const selected = Array.from(files).slice(0, Math.max(0, 3 - state.attachments.length));
  for (const file of selected) {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) continue;
    if (file.size > 5 * 1024 * 1024) {
      byId("composer-status").textContent = `${file.name} 超过 5MB，未添加`;
      continue;
    }
    const previewUrl = URL.createObjectURL(file);
    state.objectUrls.push(previewUrl);
    state.attachments.push({
      name: file.name,
      dataUrl: await readAsDataUrl(file),
      previewUrl,
    });
  }
  renderAttachmentPreview();
}

async function sendQuestion() {
  if (state.busy) return;
  const question = byId("question-input").value.trim();
  if (!question) return;
  if (!state.space) {
    byId("composer-status").textContent = "请先选择知识空间";
    return;
  }
  const sentAttachments = state.attachments.slice();
  ensureCurrentConversation(question);
  appendUserMessage(question, sentAttachments);
  recordHistoryMessage({
    role: "user",
    text: safeHistoryText(question),
    attachmentCount: sentAttachments.length,
  });
  byId("question-input").value = "";
  byId("question-input").style.height = "auto";
  state.attachments = [];
  renderAttachmentPreview();
  const pending = appendThinking();
  state.busy = true;
  byId("send-button").disabled = true;
  byId("composer-status").textContent = "正在检索已发布知识并生成回答…";
  try {
    const stream = createStreamingAnswer(pending);
    const result = await streamToolApi("/tools/v1/answer_customer_question", {
      method: "POST",
      headers: {
        "X-Knowledge-Space-Id": state.space,
        "X-Request-Id": makeId("wb_req_"),
        "X-Client-Type": `${state.mode}-workbench`,
      },
      body: JSON.stringify({
        schema_version: "1.0",
        question,
        attachments: sentAttachments.map((item) => item.dataUrl),
        conversation_context: {session_id: state.sessionId},
        response_mode: "stream",
      }),
    }, {
      status: (payload) => {
        const labels = {
          accepted: "请求已接收，正在准备客服核心…",
          understanding_and_retrieval: "正在理解问题、识别图片并检索知识库…",
          model_generating: "模型正在实时生成答案…",
          validating_answer: "正在校验答案与证据…",
        };
        byId("composer-status").textContent = labels[payload.stage] || "正在处理…";
      },
      answer_metadata: (payload) => {
        stream.imageRefs = imageReferences(payload);
        const count = stream.imageRefs.length;
        if (stream.answer) renderStreamingAnswer(stream);
        byId("composer-status").textContent = count
          ? `已找到 ${count} 张手册图片，正在生成图文答案…`
          : "检索完成，正在生成答案…";
      },
      answer_reset: () => {
        resetStreamingAnswer(stream);
      },
      answer_delta: async (payload) => {
        const previousPicCount = completePicCount(stream.answer);
        stream.answer += String(payload.text || "");
        renderStreamingAnswer(stream);
        await waitForInlineImagePaint(stream, previousPicCount);
      },
    });
    renderAnswer(result, pending, stream);
    const answerData = result.data || {};
    recordHistoryMessage({
      role: "assistant",
      text: safeHistoryText(answerData.answer || "没有返回答案。"),
      imageRefs: safeImageRefs(answerData),
      metaText: `知识版本 ${answerData.knowledge_version || "—"} · ${Math.round(Number(result.elapsed_ms) || 0)} ms`,
      traceData: safeTraceData(answerData),
      elapsedMs: Number(result.elapsed_ms) || 0,
    });
    byId("composer-status").textContent = result.data.escalation_required ? "回答完成 · 建议转人工" : "回答完成";
  } catch (error) {
    renderError(error, pending);
    byId("composer-status").textContent = "请求失败，请检查权限或知识空间";
  } finally {
    state.busy = false;
    byId("send-button").disabled = false;
    scrollMessages();
  }
}

async function loadSpaces() {
  const spaces = await api(`/control/v1/tenants/${encodeURIComponent(state.tenant)}/knowledge-spaces`);
  const select = byId("space-select");
  select.replaceChildren();
  if (!spaces.length) {
    const option = element("option", "暂无知识空间");
    option.value = "";
    select.append(option);
    state.space = "";
    byId("space-status").textContent = "请在企业控制台创建知识空间";
    return;
  }
  for (const space of spaces) {
    const option = element("option", space.name || space.knowledge_space_id);
    option.value = space.knowledge_space_id;
    select.append(option);
  }
  state.space = select.value;
  byId("space-status").textContent = `${spaces.length} 个可用空间`;
}

function loadCompetitionSpace() {
  const select = byId("space-select");
  select.replaceChildren();
  const option = element("option", "官方比赛知识库");
  option.value = "competition";
  select.append(option);
  select.disabled = true;
  state.space = "competition";
  byId("space-status").textContent = "冻结版本 · 与官方提交接口隔离兼容";
}

function applyMode(mode) {
  if (!["competition", "enterprise"].includes(mode)) return;
  state.mode = mode;
  for (const button of document.querySelectorAll(".mode-option")) {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  const enterprise = mode === "enterprise";
  byId("tenant-field").hidden = !enterprise;
  byId("tenant-input").required = enterprise;
  byId("tenant-input").disabled = !enterprise;
  byId("token-label").textContent = enterprise ? "企业 API Token" : "比赛 API Token";
  byId("token-input").placeholder = enterprise ? "rcp_…" : "输入比赛 API Token";
  byId("login-helper").textContent = enterprise
    ? "企业模式按租户隔离，只使用当前知识空间已发布的手册。"
    : "比赛演示使用冻结知识库，回答协议与官方提交接口保持一致。";
  byId("login-error").textContent = "";
}

function renderIdentity(identity) {
  const label = state.mode === "competition" ? "比赛演示" : identity.user_id;
  const role = state.mode === "competition" ? "competition" : identity.role;
  byId("identity-user").textContent = label;
  byId("identity-role").textContent = role;
  byId("identity-avatar").textContent = (label || "U").slice(0, 1).toUpperCase();
  const modeLabel = state.mode === "competition" ? "比赛演示" : "企业知识";
  byId("sidebar-mode").textContent = modeLabel;
  byId("workspace-mode-badge").textContent = modeLabel;
  byId("admin-link").hidden = state.mode !== "enterprise";
}

async function login(event) {
  event.preventDefault();
  byId("login-error").textContent = "";
  state.token = byId("token-input").value.trim();
  byId("token-input").value = "";
  try {
    let identity;
    if (state.mode === "competition") {
      await api("/tools/v1");
      state.tenant = "default";
      identity = {
        tenant_id: "default",
        user_id: "legacy-competition-api",
        role: "legacy",
        auth_type: "legacy_competition",
      };
      loadCompetitionSpace();
    } else {
      state.tenant = byId("tenant-input").value.trim();
      identity = await api("/control/v1/me");
      if (identity.tenant_id !== state.tenant) throw new Error("Token 与租户 ID 不匹配");
      byId("space-select").disabled = false;
      await loadSpaces();
    }
    state.identity = identity;
    renderIdentity(identity);
    byId("login-view").hidden = true;
    byId("workspace-view").hidden = false;
    loadConversationHistory();
    const activeSession = storageGet(activeConversationStorageKey());
    if (activeSession && state.conversations.some((item) => item.sessionId === activeSession)) {
      openConversation(activeSession);
    } else {
      resetConversation();
    }
    byId("question-input").focus();
  } catch (error) {
    state.token = "";
    byId("login-error").textContent = error instanceof Error ? error.message : String(error);
  }
}

function disconnect() {
  cleanupObjectUrls();
  state.token = "";
  state.tenant = "";
  state.identity = null;
  state.space = "";
  state.attachments = [];
  state.conversations = [];
  state.currentConversationId = "";
  state.historyKey = "";
  byId("workspace-view").hidden = true;
  byId("login-view").hidden = false;
  byId("tenant-input").value = "";
  byId("token-input").value = "";
  byId("space-select").disabled = false;
}

byId("login-form").addEventListener("submit", login);
for (const option of document.querySelectorAll(".mode-option")) {
  option.addEventListener("click", () => applyMode(option.dataset.mode));
}
byId("disconnect").addEventListener("click", disconnect);
byId("new-chat").addEventListener("click", resetConversation);
byId("space-select").addEventListener("change", (event) => {
  state.space = event.target.value;
  loadConversationHistory();
  resetConversation();
});
for (const chip of document.querySelectorAll(".prompt-chip")) {
  chip.addEventListener("click", () => {
    byId("question-input").value = chip.textContent.trim();
    byId("question-input").focus();
  });
}
byId("file-input").addEventListener("change", async (event) => {
  await acceptFiles(event.target.files || []);
  event.target.value = "";
});
byId("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuestion();
});
byId("question-input").addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    await sendQuestion();
  }
});
byId("question-input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
});

applyMode("competition");
