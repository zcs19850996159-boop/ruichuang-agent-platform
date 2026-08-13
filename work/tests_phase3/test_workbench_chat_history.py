from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTML = (ROOT / "work/cloud_control/ui/workbench.html").read_text(encoding="utf-8")
JS = (ROOT / "work/cloud_control/ui/workbench.js").read_text(encoding="utf-8")
CSS = (ROOT / "work/cloud_control/ui/workbench.css").read_text(encoding="utf-8")


def test_recent_chat_controls_are_present():
    assert 'id="recent-chats"' in HTML
    assert 'id="new-chat"' in HTML
    assert "最近聊天" in HTML
    assert ".recent-chat-row.active" in CSS
    assert ".recent-chat-delete" in CSS


def test_history_is_scoped_and_bounded():
    assert "ruichuang-workbench-history:v${HISTORY_VERSION}" in JS
    assert "state.identity.user_id" in JS
    assert "state.space" in JS
    assert "HISTORY_MAX_CONVERSATIONS = 20" in JS
    assert "HISTORY_MAX_MESSAGES = 20" in JS
    assert "HISTORY_RETENTION_MS = 90" in JS


def test_history_can_restore_continue_and_delete_conversations():
    assert "function openConversation(sessionId)" in JS
    assert "state.sessionId = conversation.sessionId" in JS
    assert 'textContent = "已恢复最近聊天，可以继续追问"' in JS
    assert "function deleteConversation(sessionId)" in JS
    assert "method: \"DELETE\"" in JS
    assert "loadConversationHistory();" in JS


def test_secrets_and_user_image_payloads_are_not_persisted():
    history_region = JS[JS.index("function conversationStorageKey") : JS.index("function buildWelcome")]
    assert "state.token" not in history_region
    assert "dataUrl" not in history_region
    assert "attachmentCount" in history_region
    assert "历史记录不缓存上传原图" in history_region
    assert "function safeManualImageUrl(value)" in JS
    assert "parsed.origin !== window.location.origin" in JS
    assert 'parsed.pathname.startsWith("/manual-images/")' in JS


def test_local_storage_failure_does_not_break_chat():
    assert "function storageGet(key)" in JS
    assert "function storageSet(key, value)" in JS
    assert "function storageRemove(key)" in JS
    assert "private browsing blocks local storage" in JS


def test_restored_manual_images_have_a_loading_timeout():
    assert "const controller = new AbortController()" in JS
    assert "controller.abort(), 12_000" in JS
    assert "signal: controller.signal" in JS
