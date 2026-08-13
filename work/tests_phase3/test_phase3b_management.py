from __future__ import annotations

import concurrent.futures
import io
import json
import sqlite3
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloud_control import ControlPlaneError, ControlPlaneService, SQLiteControlPlaneStore
from cloud_control.rest_adapter import create_control_router, control_error_handler
from knowledge_lifecycle import KnowledgeLifecycleService, KnowledgeVersionStore


PEPPER = "phase3b-test-pepper-with-at-least-32-bytes"


def bootstrap(tmp_path: Path):
    control = ControlPlaneService(
        SQLiteControlPlaneStore(tmp_path / "control.sqlite3"),
        token_pepper=PEPPER,
    )
    result = control.bootstrap_tenant(
        tenant_id="tenant-a",
        tenant_name="Tenant A",
        owner_user_id="alice",
        owner_display_name="Alice",
    )
    owner = control.authenticate(result["api_token"])
    assert owner is not None
    return control, owner


def management_client(tmp_path: Path):
    control, owner = bootstrap(tmp_path)
    control.create_knowledge_space(
        owner,
        tenant_id="tenant-a",
        space_id="manuals",
        name="Product manuals",
    )
    owner_token = next(
        result
        for result in [control.issue_token(
            owner,
            tenant_id="tenant-a",
            user_id="alice",
        )]
    )["api_token"]
    lifecycle = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "knowledge")
    )
    app = FastAPI()
    app.add_exception_handler(ControlPlaneError, control_error_handler)
    app.include_router(create_control_router(control, lifecycle))
    return TestClient(app), owner_token, control, lifecycle


def test_member_disable_is_immediate_and_final_owner_is_protected(
    tmp_path: Path,
) -> None:
    control, owner = bootstrap(tmp_path)
    control.create_member(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        display_name="Agent A",
        role="agent",
    )
    issued = control.issue_token(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
    )
    assert control.authenticate(issued["api_token"]) is not None
    updated = control.update_member(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        role="agent",
        status="disabled",
    )
    assert updated["status"] == "disabled"
    assert control.authenticate(issued["api_token"]) is None

    with pytest.raises(ControlPlaneError) as last_owner:
        control.update_member(
            owner,
            tenant_id="tenant-a",
            user_id="alice",
            role="admin",
            status="active",
        )
    assert last_owner.value.code == "last_owner_required"
    assert last_owner.value.http_status == 409


def test_disabled_member_cannot_receive_new_token(tmp_path: Path) -> None:
    control, owner = bootstrap(tmp_path)
    control.create_member(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        display_name="Agent A",
        role="agent",
    )
    control.update_member(
        owner,
        tenant_id="tenant-a",
        user_id="agent-a",
        role="agent",
        status="disabled",
    )

    with pytest.raises(ControlPlaneError) as denied:
        control.issue_token(
            owner,
            tenant_id="tenant-a",
            user_id="agent-a",
        )

    assert denied.value.code == "input_invalid"
    assert "not active" in denied.value.message


def test_concurrent_owner_demotions_preserve_one_owner(tmp_path: Path) -> None:
    control, alice = bootstrap(tmp_path)
    control.create_member(
        alice,
        tenant_id="tenant-a",
        user_id="bob",
        display_name="Bob",
        role="owner",
    )
    bob_token = control.issue_token(
        alice,
        tenant_id="tenant-a",
        user_id="bob",
    )
    bob = control.authenticate(bob_token["api_token"])
    assert bob is not None
    start = threading.Barrier(2)

    def demote(principal, user_id: str) -> tuple[str, str]:
        start.wait(timeout=5)
        try:
            control.update_member(
                principal,
                tenant_id="tenant-a",
                user_id=user_id,
                role="admin",
                status="active",
            )
            return user_id, "updated"
        except ControlPlaneError as exc:
            return user_id, exc.code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: demote(*item),
                [(alice, "alice"), (bob, "bob")],
            )
        )

    assert sorted(status for _, status in results) == [
        "last_owner_required",
        "updated",
    ]
    assert control.store.active_owner_count("tenant-a") == 1


def test_token_inventory_contains_metadata_only(tmp_path: Path) -> None:
    control, owner = bootstrap(tmp_path)
    rows = control.list_tokens(owner, tenant_id="tenant-a")
    assert rows
    serialized = json.dumps(rows)
    assert "token_hash" not in serialized
    assert "api_token" not in serialized
    assert rows[0]["token_prefix"].startswith("rcp_")


def test_admin_cannot_change_owner_membership(tmp_path: Path) -> None:
    control, owner = bootstrap(tmp_path)
    control.create_member(
        owner,
        tenant_id="tenant-a",
        user_id="admin-a",
        display_name="Admin A",
        role="admin",
    )
    issued = control.issue_token(
        owner,
        tenant_id="tenant-a",
        user_id="admin-a",
    )
    admin = control.authenticate(issued["api_token"])
    assert admin is not None
    with pytest.raises(ControlPlaneError) as denied:
        control.update_member(
            admin,
            tenant_id="tenant-a",
            user_id="alice",
            role="admin",
            status="active",
        )
    assert denied.value.http_status == 403


def test_existing_phase3a_database_migrates_membership_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-control.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE tenants (
                tenant_id TEXT PRIMARY KEY, name TEXT, status TEXT, created_at TEXT
            );
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY, display_name TEXT, status TEXT, created_at TEXT
            );
            CREATE TABLE memberships (
                tenant_id TEXT, user_id TEXT, role TEXT, created_at TEXT,
                PRIMARY KEY (tenant_id, user_id)
            );
            """
        )
    SQLiteControlPlaneStore(path)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memberships)")
        }
        version = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    assert "status" in columns
    assert version == "3"


def test_knowledge_version_inventory_is_scoped_and_path_free(
    tmp_path: Path,
) -> None:
    lifecycle = KnowledgeLifecycleService(
        KnowledgeVersionStore(tmp_path / "knowledge")
    )
    manual = tmp_path / "manual.txt"
    manual.write_text(
        "X300 开机方法\n按住电源键三秒启动设备。",
        encoding="utf-8",
    )
    staged = lifecycle.ingest(
        str(manual),
        product_id="x300",
        actor="alice",
        tenant_id="tenant-a",
        space_id="manuals",
    )
    lifecycle.evaluate(staged["staging_id"])
    lifecycle.publish(
        staged["staging_id"],
        tenant_id="tenant-a",
        space_id="manuals",
        version="knowledge-v1",
        approved_by="alice",
    )

    versions = lifecycle.list_versions(
        tenant_id="tenant-a",
        space_id="manuals",
    )
    assert len(versions) == 1
    assert versions[0]["active"] is True
    assert versions[0]["document"]["original_name"] == "manual.txt"
    assert "path" not in json.dumps(versions)
    assert lifecycle.list_versions(
        tenant_id="tenant-b",
        space_id="manuals",
    ) == []
    assert lifecycle.active_version(
        tenant_id="tenant-a",
        space_id="manuals",
    )["version"] == "knowledge-v1"


def test_admin_ui_keeps_credentials_in_memory_and_avoids_html_injection() -> None:
    ui_root = Path(__file__).resolve().parents[1] / "cloud_control" / "ui"
    script = (ui_root / "admin.js").read_text(encoding="utf-8")
    html = (ui_root / "admin_ui.html").read_text(encoding="utf-8")
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "innerHTML" not in script
    assert "api_token" not in html
    assert 'type="password"' in html
    assert 'id="ingestion-form"' in html
    assert 'id="staging-list"' in html
    assert 'id="staging-lookup-form"' in html
    assert 'id="evaluate-button"' in html
    assert 'id="publish-button"' in html
    assert "/ingestions" in script
    assert "/package-ingestions" in script
    assert "/staging?limit=100" in script
    assert "inspectStaging" in script
    assert "/evaluate" in script
    assert "/regression" in script
    assert "/diagnosis" in script
    assert "/publish" in script
    assert "/rollback" in script
    assert "approved_by" not in script


def test_browser_release_flow_uploads_evaluates_publishes_and_rolls_back(
    tmp_path: Path,
) -> None:
    client, token, _, lifecycle = management_client(tmp_path)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Filename": "x300-manual.txt",
        "X-Product-Id": "x300",
        "Content-Type": "application/octet-stream",
    }
    upload = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/ingestions",
        headers=headers,
        content="X300 电池更换\n打开后盖，按正负极标记更换电池。".encode(),
    )
    assert upload.status_code == 201
    assert (
        upload.json()["data"]["manifest"]["document"]["original_name"]
        == "x300-manual.txt"
    )
    staging_v1 = upload.json()["data"]["staging_id"]

    evaluated = client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_v1}/evaluate",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["data"]["publishable"] is True

    published = client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_v1}/publish",
        headers={"Authorization": f"Bearer {token}"},
        json={"version": "knowledge-v1", "approved_by": "mallory"},
    )
    assert published.status_code == 200
    assert published.json()["data"]["manifest"]["approved_by"] == "alice"

    second_upload = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/ingestions",
        headers=headers,
        content="X300 电池更换\n关闭电源后打开后盖并更换电池。".encode(),
    )
    staging_v2 = second_upload.json()["data"]["staging_id"]
    client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_v2}/evaluate",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_v2}/publish",
        headers={"Authorization": f"Bearer {token}"},
        json={"version": "knowledge-v2"},
    )
    assert lifecycle.active_version(
        tenant_id="tenant-a",
        space_id="manuals",
    )["version"] == "knowledge-v2"

    rolled_back = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/versions/knowledge-v1/rollback",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["data"]["version"] == "knowledge-v1"


def test_staging_inventory_is_visible_scoped_and_path_free(
    tmp_path: Path,
) -> None:
    client, token, control, _ = management_client(tmp_path)
    owner = control.authenticate(token)
    assert owner is not None
    control.create_knowledge_space(
        owner,
        tenant_id="tenant-a",
        space_id="private",
        name="Private manuals",
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Filename": "x300-manual.txt",
        "X-Product-Id": "x300",
        "Content-Type": "application/octet-stream",
    }
    uploaded = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/ingestions",
        headers=headers,
        content="X300 启动方法\n长按电源键三秒。".encode(),
    )
    staging_id = uploaded.json()["data"]["staging_id"]

    inventory = client.get(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert inventory.status_code == 200
    assert [row["staging_id"] for row in inventory.json()["data"]] == [
        staging_id
    ]
    row = inventory.json()["data"][0]
    assert row["status"] == "staged"
    assert row["quality"]["publishable"] is None
    assert row["regression"]["status"] is None
    assert row["created_at"]
    assert "path" not in json.dumps(row)

    exact = client.get(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert exact.status_code == 200
    assert exact.json()["data"]["created_at"] == row["created_at"]

    private_inventory = client.get(
        "/control/v1/tenants/tenant-a/knowledge-spaces/private/staging",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert private_inventory.status_code == 200
    assert private_inventory.json()["data"] == []


def test_product_package_api_runs_regression_diagnosis_and_quality(
    tmp_path: Path,
) -> None:
    client, token, _, _ = management_client(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "product_id": "chair-x",
                    "models": ["CX100"],
                    "files": [
                        {
                            "path": "documents/manual.txt",
                            "role": "manual",
                            "language": "zh-CN",
                            "model": "CX100",
                            "version": "v1",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        )
        archive.writestr(
            "documents/manual.txt",
            json.dumps(
                [
                    "CX100 更换电池前关闭电源。<PIC>",
                    ["CX100_battery"],
                ],
                ensure_ascii=False,
            ),
        )
        archive.writestr("images/CX100_battery.png", b"image")
    auth = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/zip",
    }
    uploaded = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/package-ingestions",
        headers=auth,
        content=buffer.getvalue(),
    )
    assert uploaded.status_code == 201
    staging_id = uploaded.json()["data"]["staging_id"]
    assert (
        uploaded.json()["data"]["manifest"]["package"]["document_count"]
        == 1
    )

    regression = client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_id}/regression",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "cases": [
                {
                    "question": "CX100 如何更换电池",
                    "expected_product_id": "chair-x",
                    "required_terms": ["关闭电源"],
                    "expected_image_ids": ["CX100_battery"],
                }
            ]
        },
    )
    assert regression.status_code == 200
    assert regression.json()["data"]["status"] == "passed"

    quality = client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_id}/evaluate",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert quality.status_code == 200
    assert quality.json()["data"]["publishable"] is True
    assert (
        quality.json()["data"]["retrieval"]["strategy_version"]
        == "ruichuang-managed-hybrid-v2"
    )
    diagnosis = client.get(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/manuals/staging/{staging_id}/diagnosis",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert diagnosis.status_code == 200
    assert diagnosis.json()["data"]["next_action"] == "request_explicit_publish_approval"


def test_browser_release_flow_rejects_unsupported_and_cross_space_staging(
    tmp_path: Path,
) -> None:
    client, token, control, _ = management_client(tmp_path)
    owner = control.authenticate(token)
    assert owner is not None
    control.create_knowledge_space(
        owner,
        tenant_id="tenant-a",
        space_id="private",
        name="Private manuals",
    )
    auth = {"Authorization": f"Bearer {token}"}
    unsupported = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/ingestions",
        headers={
            **auth,
            "X-Filename": "manual.exe",
            "X-Product-Id": "x300",
        },
        content=b"not a document",
    )
    assert unsupported.status_code == 400

    uploaded = client.post(
        "/control/v1/tenants/tenant-a/knowledge-spaces/manuals/ingestions",
        headers={
            **auth,
            "X-Filename": "manual.txt",
            "X-Product-Id": "x300",
        },
        content="X300 操作方法".encode(),
    )
    staging_id = uploaded.json()["data"]["staging_id"]
    cross_space = client.post(
        f"/control/v1/tenants/tenant-a/knowledge-spaces/private/staging/{staging_id}/evaluate",
        headers=auth,
        json={},
    )
    assert cross_space.status_code == 404


def test_workbench_unifies_profiles_without_persisting_credentials() -> None:
    ui_root = Path(__file__).resolve().parents[1] / "cloud_control" / "ui"
    script = (ui_root / "workbench.js").read_text(encoding="utf-8")
    html = (ui_root / "workbench.html").read_text(encoding="utf-8")

    history_region = script[
        script.index("function conversationStorageKey") : script.index("function buildWelcome")
    ]
    assert "localStorage" in script
    assert "state.token" not in history_region
    assert "dataUrl" not in history_region
    assert 'byId("token-input").value = ""' in script
    assert "sessionStorage" not in script
    assert "innerHTML" not in script
    assert script.count('"/tools/v1/answer_customer_question"') == 1
    assert 'api("/tools/v1")' in script
    assert 'state.mode === "competition"' in script
    assert 'option.value = "competition"' in script
    assert '"/control/v1/me"' in script
    assert '"/chat"' not in script
    assert "data.core_result?.selector?.confidence?.score" in script
    assert 'typeof data.validation?.pass === "boolean"' in script
    assert 'response_mode: "stream"' in script
    assert 'Accept: "text/event-stream"' in script
    assert "response.body.getReader()" in script
    assert "answer_delta:" in script
    assert "answer_reset:" in script
    assert "createStreamingAnswer" in script
    assert "visibleStreamingAnswer" in script
    assert "stream.imageRefs = imageReferences(payload)" in script
    assert "nodes.push(streamingFigure(stream, index))" in script
    assert "await handlers[event]?.(payload)" in script
    assert "waitForInlineImagePaint" in script
    assert "Promise.allSettled(imageLoads)" in script
    assert "requestAnimationFrame" in script
    assert 'replaceAll("<PIC>", "")' not in script
    assert '"X-Knowledge-Space-Id"' in script
    assert "attachments:" in script
    assert 'type="password"' in html
    assert 'data-mode="competition"' in html
    assert 'data-mode="enterprise"' in html
    assert 'id="trace-profile"' in html
    assert 'id="trace-space"' in html
    assert 'src="/control-ui/workbench.js"' in html
    assert 'href="/control-ui/workbench.css"' in html
    assert "<script>" not in html
    assert "<style>" not in html


def test_workbench_route_has_external_assets_and_strict_csp() -> None:
    server = (
        Path(__file__).resolve().parents[1] / "fastapi_server.py"
    ).read_text(encoding="utf-8")

    assert '@app.get("/workbench"' in server
    assert '"workbench.html"' in server
    assert '"workbench.css"' in server
    assert '"workbench.js"' in server
    assert "img-src 'self' data: blob:" in server
    assert "frame-ancestors 'none'" in server
