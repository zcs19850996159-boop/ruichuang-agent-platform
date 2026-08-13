"use strict";

const state = {
  token: "",
  tenant: "",
  identity: null,
  selectedSpace: "",
  selectedSpaceName: "",
  stagingId: "",
  staging: null,
  quality: null,
};
const byId = (id) => document.getElementById(id);

function text(value) {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function node(tag, value, className) {
  const element = document.createElement(tag);
  if (value !== undefined) element.textContent = text(value);
  if (className) element.className = className;
  return element;
}

function table(columns, rows, actions) {
  if (!rows.length) return node("p", "暂无数据", "empty");
  const result = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) headRow.append(node("th", column.label));
  if (actions) headRow.append(node("th", "操作"));
  head.append(headRow);
  result.append(head);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const line = document.createElement("tr");
    for (const column of columns) line.append(node("td", column.value(row)));
    if (actions) {
      const actionCell = node("td", undefined, "actions");
      for (const action of actions(row)) actionCell.append(action);
      line.append(actionCell);
    }
    body.append(line);
  }
  result.append(body);
  return result;
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error?.message || payload.detail || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload.data;
}

function endpoint(suffix) {
  return `/control/v1/tenants/${encodeURIComponent(state.tenant)}${suffix}`;
}

function showError(error) {
  byId("workspace-error").textContent = error instanceof Error ? error.message : String(error);
}

function setReleaseStatus(message, kind = "") {
  const status = byId("release-status");
  status.textContent = message;
  status.className = `operation-status ${kind}`.trim();
}

function setReleaseStep(step) {
  const order = ["upload", "evaluate", "publish"];
  for (const [index, name] of order.entries()) {
    const item = byId(`step-${name}`);
    item.classList.toggle("complete", index < step);
    item.classList.toggle("current", index === step);
  }
}

function resetRelease() {
  state.stagingId = "";
  state.staging = null;
  state.quality = null;
  byId("release-result").hidden = true;
  byId("quality-metrics").replaceChildren();
  byId("publish-button").disabled = true;
  byId("diagnosis-button").disabled = true;
  byId("regression-button").disabled = true;
  setReleaseStep(0);
  if (state.selectedSpace) {
    byId("release-space").textContent = state.selectedSpaceName
      ? `${state.selectedSpaceName} · ${state.selectedSpace}`
      : state.selectedSpace;
    setReleaseStatus("已选择目标空间，可以上传手册或商品资料包。", "success");
  } else {
    byId("release-space").textContent = "未选择知识空间";
    setReleaseStatus("请先在上方知识空间列表中选择目标空间。");
  }
}

function chooseSpace(space) {
  state.selectedSpace = space.knowledge_space_id;
  state.selectedSpaceName = space.name;
  resetRelease();
  Promise.all([loadVersions(space), loadStaging(space)]).catch(showError);
  byId("ingestion-product").focus();
}

async function loadMembers() {
  const rows = await api(endpoint("/members"));
  const actionFactory = (member) => {
    const toggle = node("button", member.status === "active" ? "停用" : "启用", member.status === "active" ? "danger" : "ghost");
    toggle.type = "button";
    toggle.addEventListener("click", async () => {
      try {
        await api(endpoint(`/members/${encodeURIComponent(member.user_id)}`), {
          method: "PATCH",
          body: JSON.stringify({
            role: member.role,
            status: member.status === "active" ? "disabled" : "active",
          }),
        });
        await Promise.all([loadMembers(), loadAudit()]);
      } catch (error) { showError(error); }
    });
    return [toggle];
  };
  byId("members").replaceChildren(table([
    { label: "用户", value: (row) => row.user_id },
    { label: "名称", value: (row) => row.display_name },
    { label: "角色", value: (row) => row.role },
    { label: "状态", value: (row) => row.status },
  ], rows, actionFactory));
}

async function loadSpaces() {
  const rows = await api(endpoint("/knowledge-spaces"));
  const actionFactory = (space) => {
    const inspect = node(
      "button",
      state.selectedSpace === space.knowledge_space_id ? "已选择" : "选择并管理",
      state.selectedSpace === space.knowledge_space_id ? "" : "ghost",
    );
    inspect.type = "button";
    inspect.addEventListener("click", () => chooseSpace(space));
    return [inspect];
  };
  byId("spaces").replaceChildren(table([
    { label: "空间", value: (row) => row.knowledge_space_id },
    { label: "名称", value: (row) => row.name },
    { label: "状态", value: (row) => row.status },
  ], rows, actionFactory));
}

async function loadVersions(space) {
  state.selectedSpace = space.knowledge_space_id;
  state.selectedSpaceName = space.name;
  byId("versions-title").textContent = `${space.name} · 版本`;
  const rows = await api(endpoint(`/knowledge-spaces/${encodeURIComponent(space.knowledge_space_id)}/versions`));
  const actionFactory = (version) => {
    const rollback = node("button", version.active ? "当前生效" : "回滚到此版本", version.active ? "ghost" : "danger");
    rollback.type = "button";
    rollback.disabled = version.active;
    rollback.addEventListener("click", async () => {
      if (!window.confirm(`确认将 ${space.name} 回滚到 ${version.version}？`)) return;
      try {
        rollback.disabled = true;
        setReleaseStatus(`正在回滚到 ${version.version}…`);
        await api(endpoint(
          `/knowledge-spaces/${encodeURIComponent(space.knowledge_space_id)}/versions/${encodeURIComponent(version.version)}/rollback`,
        ), {method: "POST", body: JSON.stringify({})});
        setReleaseStatus(`已回滚到 ${version.version}。`, "success");
        await Promise.all([loadVersions(space), loadAudit()]);
      } catch (error) {
        rollback.disabled = false;
        setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
      }
    });
    return [rollback];
  };
  byId("versions").replaceChildren(table([
    { label: "版本", value: (row) => row.version },
    { label: "Active", value: (row) => row.active ? "是" : "否" },
    { label: "文档", value: (row) => row.document?.original_name },
    { label: "Chunks", value: (row) => row.chunk_count },
    { label: "发布时间", value: (row) => row.published_at },
  ], rows, actionFactory));
}

function regressionLabel(regression) {
  if (!regression?.status) return "未运行";
  if (regression.status === "passed") {
    return `通过 ${regression.passed}/${regression.total}`;
  }
  return `失败 ${regression.failed}/${regression.total}`;
}

function stagingStatusLabel(row) {
  if (row.quality?.publishable === true) return "待批准发布";
  if (row.quality?.publishable === false) return "质量门禁阻止";
  return "待质量检查";
}

async function inspectStaging(stagingId) {
  if (!state.selectedSpace) throw new Error("请先选择知识空间");
  const data = await api(endpoint(
    `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/staging/${encodeURIComponent(stagingId)}`,
  ));
  renderStaging(data);
  if (data.quality) renderQuality(data.quality);
  setReleaseStatus(
    data.quality?.publishable === true
      ? "已载入历史暂存版本：质量门禁通过，等待人工批准发布。"
      : "已载入历史暂存版本，可以继续诊断、回归或质量检查。",
    data.quality?.publishable === true ? "success" : "",
  );
  byId("release-result").scrollIntoView({behavior: "smooth", block: "start"});
}

async function loadStaging(space) {
  state.selectedSpace = space.knowledge_space_id;
  state.selectedSpaceName = space.name;
  byId("staging-title").textContent = `${space.name} · 暂存版本`;
  const rows = await api(endpoint(
    `/knowledge-spaces/${encodeURIComponent(space.knowledge_space_id)}/staging?limit=100`,
  ));
  const actionFactory = (row) => {
    const inspect = node("button", "查看与继续", "ghost");
    inspect.type = "button";
    inspect.addEventListener("click", async () => {
      try {
        inspect.disabled = true;
        await inspectStaging(row.staging_id);
      } catch (error) {
        showError(error);
      } finally {
        inspect.disabled = false;
      }
    });
    return [inspect];
  };
  byId("staging-list").replaceChildren(table([
    {label: "暂存 ID", value: (row) => row.staging_id},
    {label: "产品", value: (row) => row.product_id},
    {label: "文档", value: (row) => row.document?.original_name},
    {label: "创建时间", value: (row) => row.created_at},
    {label: "质量状态", value: stagingStatusLabel},
    {label: "知识回归", value: (row) => regressionLabel(row.regression)},
  ], rows, actionFactory));
}

function renderStaging(data) {
  state.staging = data;
  state.stagingId = data.staging_id;
  byId("release-result").hidden = false;
  byId("staging-id").textContent = data.staging_id;
  const documents = data.manifest?.documents || data.documents || [];
  byId("staging-document").textContent = documents.length > 1
    ? `${documents.length} 份资料`
    : data.manifest?.document?.original_name
    || data.document?.original_name
    || "—";
  byId("staging-chunks").textContent = data.manifest?.chunks ?? data.chunk_count ?? "—";
  byId("staging-images").textContent = data.manifest?.images?.length ?? data.image_count ?? "—";
  byId("quality-status").textContent = "等待检查";
  byId("quality-status").className = "status-pending";
  byId("quality-blockers").textContent = "解析完成，尚未批准发布；请先执行入库质量检查。";
  byId("quality-blockers").className = "quality-message muted";
  byId("quality-metrics").replaceChildren();
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
  ].join("");
  byId("publish-version").value = `knowledge-v${stamp}`;
  byId("publish-button").disabled = true;
  byId("diagnosis-button").disabled = false;
  byId("regression-button").disabled = false;
  setReleaseStep(1);
}

function renderQuality(report) {
  state.quality = report;
  const labels = [
    ["文档", report.metrics?.document_count],
    ["文本块", report.metrics?.chunk_count],
    ["图片", report.metrics?.image_count],
    ["缺失图片", report.metrics?.missing_image_count],
    ["重复率", `${Math.round(Number(report.metrics?.duplicate_chunk_ratio || 0) * 100)}%`],
    ["检索", report.retrieval?.mode || "—"],
  ];
  byId("quality-metrics").replaceChildren(...labels.map(([label, value]) => {
    const item = node("div");
    item.append(node("span", label), node("strong", value));
    return item;
  }));
  const publishable = report.publishable === true;
  byId("quality-status").textContent = publishable ? "允许发布" : "阻止发布";
  byId("quality-status").className = publishable ? "status-ok" : "status-error";
  const blockers = report.blockers || [];
  byId("quality-blockers").textContent = publishable
    ? "质量、安全与图片完整性门禁均已通过，可以人工批准发布。"
    : `阻断项：${blockers.join("、") || "未知质量问题"}`;
  byId("quality-blockers").className = `quality-message ${publishable ? "success" : "error"}`;
  byId("publish-button").disabled = !publishable;
  setReleaseStep(publishable ? 2 : 1);
}

async function loadTokens() {
  const rows = await api(endpoint("/tokens"));
  const actionFactory = (token) => {
    const revoke = node("button", "撤销", "danger");
    revoke.type = "button";
    revoke.disabled = token.status !== "active";
    revoke.addEventListener("click", async () => {
      try {
        await api(endpoint(`/tokens/${encodeURIComponent(token.token_id)}`), { method: "DELETE" });
        await Promise.all([loadTokens(), loadAudit()]);
      } catch (error) { showError(error); }
    });
    return [revoke];
  };
  byId("tokens").replaceChildren(table([
    { label: "Token ID", value: (row) => row.token_id },
    { label: "用户", value: (row) => row.user_id },
    { label: "前缀", value: (row) => row.token_prefix },
    { label: "状态", value: (row) => row.status },
    { label: "最近使用", value: (row) => row.last_used_at },
  ], rows, actionFactory));
}

async function loadAudit() {
  const rows = await api(endpoint("/audit?limit=100"));
  byId("audit").replaceChildren(table([
    { label: "时间", value: (row) => row.timestamp },
    { label: "操作者", value: (row) => row.actor_user_id },
    { label: "动作", value: (row) => row.action },
    { label: "资源", value: (row) => `${row.resource_type}:${row.resource_id}` },
    { label: "结果", value: (row) => row.outcome },
  ], rows));
}

async function refreshAll() {
  byId("workspace-error").textContent = "";
  await Promise.all([loadMembers(), loadSpaces(), loadTokens(), loadAudit()]);
}

byId("refresh-staging").addEventListener("click", async () => {
  if (!state.selectedSpace) {
    showError(new Error("请先选择知识空间"));
    return;
  }
  try {
    await loadStaging({
      knowledge_space_id: state.selectedSpace,
      name: state.selectedSpaceName || state.selectedSpace,
    });
  } catch (error) {
    showError(error);
  }
});

byId("staging-lookup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await inspectStaging(byId("staging-lookup-id").value.trim());
  } catch (error) {
    showError(error);
  }
});

byId("ingestion-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedSpace) {
    setReleaseStatus("请先选择目标知识空间。", "error");
    return;
  }
  const file = byId("ingestion-file").files[0];
  if (!file) {
    setReleaseStatus("请选择手册或商品资料包。", "error");
    return;
  }
  const packageUpload = file.name.toLowerCase().endsWith(".zip");
  const productId = byId("ingestion-product").value.trim();
  if (!packageUpload && !productId) {
    setReleaseStatus("单文件上传必须填写产品 ID；ZIP 资料包从 manifest 读取。", "error");
    return;
  }
  const submit = byId("ingestion-submit");
  submit.disabled = true;
  try {
    setReleaseStatus(packageUpload
      ? "正在上传商品资料包，由平台统一解析、分块、绑定图片并构建索引…"
      : "正在上传、解析并绑定手册图片…");
    const ingestionPath = packageUpload
      ? `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/package-ingestions`
      : `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/ingestions`;
    const data = await api(endpoint(ingestionPath), {
      method: "POST",
      headers: packageUpload ? {
        "Content-Type": "application/zip",
      } : {
        "Content-Type": "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name),
        "X-Product-Id": productId,
      },
      body: file,
    });
    renderStaging(data);
    setReleaseStatus("商品知识已进入隔离暂存区，当前生效知识未改变。", "success");
    await loadStaging({
      knowledge_space_id: state.selectedSpace,
      name: state.selectedSpaceName || state.selectedSpace,
    });
  } catch (error) {
    setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    submit.disabled = false;
  }
});

byId("diagnosis-button").addEventListener("click", async () => {
  if (!state.stagingId || !state.selectedSpace) return;
  const button = byId("diagnosis-button");
  button.disabled = true;
  try {
    setReleaseStatus("正在诊断入库阻断项…");
    const diagnosis = await api(endpoint(
      `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/staging/${encodeURIComponent(state.stagingId)}/diagnosis`,
    ));
    const details = (diagnosis.blockers || []).map((item) => `${item.code}：${item.message}`);
    if (diagnosis.missing_image_ids?.length) {
      details.push(`缺失图片 ID：${diagnosis.missing_image_ids.join("、")}`);
    }
    byId("quality-blockers").textContent = diagnosis.publishable
      ? "当前暂存版本无阻断项，可以进入人工审批。"
      : details.join("；") || "未识别到可自动解释的阻断项，请人工检查。";
    byId("quality-blockers").className = `quality-message ${diagnosis.publishable ? "success" : "error"}`;
    setReleaseStatus(
      diagnosis.publishable ? "诊断完成：无阻断项。" : `诊断完成：下一步 ${diagnosis.next_action}。`,
      diagnosis.publishable ? "success" : "error",
    );
  } catch (error) {
    setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    button.disabled = false;
  }
});

byId("regression-button").addEventListener("click", async () => {
  if (!state.stagingId || !state.selectedSpace) return;
  const file = byId("regression-file").files[0];
  if (!file) {
    setReleaseStatus("请选择知识版本回归用例 JSON；这不是比赛评分文件。", "error");
    return;
  }
  const button = byId("regression-button");
  button.disabled = true;
  try {
    const decoded = JSON.parse(await file.text());
    const cases = Array.isArray(decoded) ? decoded : decoded.cases;
    if (!Array.isArray(cases) || !cases.length) throw new Error("回归用例 JSON 必须是数组或包含 cases 数组");
    setReleaseStatus("正在比较暂存知识与当前活动版本…");
    const report = await api(endpoint(
      `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/staging/${encodeURIComponent(state.stagingId)}/regression`,
    ), {method: "POST", body: JSON.stringify({cases})});
    byId("quality-blockers").textContent = report.failed
      ? `知识回归失败 ${report.failed}/${report.total}；分类：${Object.entries(report.failure_categories || {}).map(([name, count]) => `${name}=${count}`).join("、")}`
      : `知识回归通过 ${report.passed}/${report.total}；请重新执行入库质量检查。`;
    byId("quality-blockers").className = `quality-message ${report.failed ? "error" : "success"}`;
    setReleaseStatus(
      report.failed ? "知识版本回归未通过，禁止发布。" : "知识版本回归通过，请重新执行入库质量检查。",
      report.failed ? "error" : "success",
    );
    byId("publish-button").disabled = true;
    await loadStaging({
      knowledge_space_id: state.selectedSpace,
      name: state.selectedSpaceName || state.selectedSpace,
    });
  } catch (error) {
    setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    button.disabled = false;
  }
});

byId("evaluate-button").addEventListener("click", async () => {
  if (!state.stagingId || !state.selectedSpace) return;
  const button = byId("evaluate-button");
  button.disabled = true;
  try {
    setReleaseStatus("正在执行入库质量、安全、重复内容和图片完整性检查…");
    const report = await api(endpoint(
      `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/staging/${encodeURIComponent(state.stagingId)}/evaluate`,
    ), {method: "POST", body: JSON.stringify({})});
    renderQuality(report);
    setReleaseStatus(
      report.publishable ? "质量门禁通过，等待人工批准发布。" : "质量门禁未通过，当前版本保持不变。",
      report.publishable ? "success" : "error",
    );
    await loadAudit();
    await loadStaging({
      knowledge_space_id: state.selectedSpace,
      name: state.selectedSpaceName || state.selectedSpace,
    });
  } catch (error) {
    setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    button.disabled = false;
  }
});

byId("publish-button").addEventListener("click", async () => {
  if (!state.stagingId || !state.selectedSpace || state.quality?.publishable !== true) return;
  const version = byId("publish-version").value.trim();
  if (!version) {
    setReleaseStatus("请输入不可变发布版本号。", "error");
    return;
  }
  const button = byId("publish-button");
  button.disabled = true;
  try {
    setReleaseStatus(`正在批准并激活 ${version}…`);
    await api(endpoint(
      `/knowledge-spaces/${encodeURIComponent(state.selectedSpace)}/staging/${encodeURIComponent(state.stagingId)}/publish`,
    ), {method: "POST", body: JSON.stringify({version})});
    setReleaseStep(3);
    setReleaseStatus(`${version} 已发布并生效；旧版本仍保留，可随时回滚。`, "success");
    await Promise.all([
      loadVersions({
        knowledge_space_id: state.selectedSpace,
        name: state.selectedSpaceName || state.selectedSpace,
      }),
      loadStaging({
        knowledge_space_id: state.selectedSpace,
        name: state.selectedSpaceName || state.selectedSpace,
      }),
      loadAudit(),
    ]);
  } catch (error) {
    button.disabled = false;
    setReleaseStatus(error instanceof Error ? error.message : String(error), "error");
  }
});

byId("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  byId("login-error").textContent = "";
  state.token = byId("token-input").value.trim();
  state.tenant = byId("tenant-input").value.trim();
  byId("token-input").value = "";
  try {
    const identity = await api("/control/v1/me");
    if (identity.tenant_id !== state.tenant) throw new Error("Token 与租户 ID 不匹配");
    state.identity = identity;
    byId("identity-tenant").textContent = identity.tenant_id;
    byId("identity-user").textContent = identity.user_id;
    byId("identity-role").textContent = identity.role;
    byId("login-panel").hidden = true;
    byId("workspace").hidden = false;
    byId("disconnect").hidden = false;
    await refreshAll();
  } catch (error) {
    state.token = "";
    byId("login-error").textContent = error instanceof Error ? error.message : String(error);
  }
});

byId("disconnect").addEventListener("click", () => {
  state.token = "";
  state.tenant = "";
  state.identity = null;
  state.selectedSpace = "";
  state.selectedSpaceName = "";
  state.stagingId = "";
  state.staging = null;
  state.quality = null;
  byId("workspace").hidden = true;
  byId("disconnect").hidden = true;
  byId("login-panel").hidden = false;
  byId("issued-token-value").textContent = "";
  byId("issued-token").hidden = true;
  for (const id of ["members", "spaces", "versions", "staging-list", "tokens", "audit"]) {
    byId(id).replaceChildren();
  }
  resetRelease();
});

byId("member-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api(endpoint("/members"), {
      method: "POST",
      body: JSON.stringify({
        user_id: byId("member-id").value.trim(),
        display_name: byId("member-name").value.trim(),
        role: byId("member-role").value,
      }),
    });
    event.target.reset();
    await Promise.all([loadMembers(), loadAudit()]);
  } catch (error) { showError(error); }
});

byId("space-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api(endpoint("/knowledge-spaces"), {
      method: "POST",
      body: JSON.stringify({
        knowledge_space_id: byId("space-id").value.trim(),
        name: byId("space-name").value.trim(),
      }),
    });
    event.target.reset();
    await Promise.all([loadSpaces(), loadAudit()]);
  } catch (error) { showError(error); }
});

byId("token-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const issued = await api(endpoint("/tokens"), {
      method: "POST",
      body: JSON.stringify({ user_id: byId("token-user").value.trim() }),
    });
    byId("issued-token-value").textContent = issued.api_token;
    byId("issued-token").hidden = false;
    event.target.reset();
    await Promise.all([loadTokens(), loadAudit()]);
  } catch (error) { showError(error); }
});

byId("refresh-members").addEventListener("click", () => loadMembers().catch(showError));
byId("refresh-spaces").addEventListener("click", () => loadSpaces().catch(showError));
byId("refresh-tokens").addEventListener("click", () => loadTokens().catch(showError));
byId("refresh-audit").addEventListener("click", () => loadAudit().catch(showError));
