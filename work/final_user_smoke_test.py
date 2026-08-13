from __future__ import annotations

import base64
import json
import re
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any


API_URL = "http://127.0.0.1:6006/chat"
ENV_PATH = Path("/root/customer_agent_deploy/app/.env")


def read_token() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("KAFU_API_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("KAFU_API_TOKEN not found")


def red_png_data_url() -> str:
    width, height = 40, 30
    raw = b"".join(b"\x00" + bytes((220, 30, 30)) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def call_chat(token: str, question: str, *, session_id: str, images: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "question": question,
        "session_id": session_id,
        "images": images or [],
        "stream": False,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return body


CASES: list[dict[str, Any]] = [
    {"id": "S01", "type": "客服", "q": "你好，你能帮助我什么？", "routes": {"smalltalk"}},
    {"id": "S02", "type": "客服", "q": "我想申请退款，一般多久到账？", "routes": {"policy_service"}},
    {"id": "S03", "type": "客服", "q": "商品收到破损了，我需要提供什么凭证？", "routes": {"policy_service"}},
    {"id": "S04", "type": "客服", "q": "发票开错了怎么办，可以重开吗？", "routes": {"policy_service"}},
    {"id": "S05", "type": "客服", "q": "物流一直显示待揽收，是什么原因？", "routes": {"policy_service"}},
    {"id": "S06", "type": "客服", "q": "七天无理由退货有什么要求？", "routes": {"policy_service"}},
    {"id": "S07", "type": "客服", "q": "换货需要我先寄回商品吗？", "routes": {"policy_service"}},
    {"id": "S08", "type": "客服", "q": "商家发错货了怎么处理？", "routes": {"policy_service"}},
    {"id": "S09", "type": "客服", "q": "退货运费由谁承担？", "routes": {"policy_service"}},
    {"id": "S10", "type": "客服", "q": "售后一直没人处理，我应该怎么投诉？", "routes": {"policy_service"}},
    {"id": "O01", "type": "越界", "q": "今天中午天气怎么样？", "routes": {"out_of_scope", "smalltalk"}},
    {"id": "C01", "type": "中文手册", "q": "空调遥控器没电了，怎么更换电池？", "manual": "Manual01"},
    {"id": "C02", "type": "中文手册", "q": "如何使用空调的自清洁运行功能？如何安装空调遥控器支架？", "manual": "Manual01"},
    {"id": "C03", "type": "中文手册", "q": "空气净化器滤网应该怎么清洁？", "manual": "Manual03"},
    {"id": "C04", "type": "中文手册", "q": "电钻充电器指示灯闪烁分别是什么意思？", "manual": "Manual11"},
    {"id": "C05", "type": "中文手册", "q": "健身单车控制台主要有哪些功能？", "manual": "Manual14"},
    {"id": "C06", "type": "中文手册", "q": "健身追踪器表带有没有其他尺寸可以选？", "manual": "Manual16"},
    {"id": "C07", "type": "中文手册", "q": "发电机启动前需要检查哪些内容？", "manual": "Manual18"},
    {"id": "C08", "type": "中文手册", "q": "洗碗机上喷淋臂怎么清洁？", "manual": "Manual06"},
    {"id": "C09", "type": "中文手册", "q": "烤箱顶部加热元件怎么移动？", "manual": "Manual28"},
    {"id": "C10", "type": "中文手册", "q": "VR头显遮光罩怎么清洁？", "manual": "Manual38"},
    {"id": "C11", "type": "中文多手册", "q": "空调遥控器怎么换电池？健身追踪器表带尺寸怎么选？", "manual_contains": {"Manual01", "Manual16"}},
    {"id": "E01", "type": "英文手册", "q": "In the camera manual, how do I change AF mode?", "manual": "Manual10", "english": True},
    {"id": "E02", "type": "英文手册", "q": "In the boat manual, how do I install the anchor light?", "manual": "Manual09", "english": True},
    {"id": "E03", "type": "英文手册", "q": "In the fax manual, what safety precautions should I read before use?", "manual": "Manual15", "english": True},
    {"id": "E04", "type": "英文手册", "q": "In the microwave manual, how do I replace the charcoal filter?", "manual": "Manual24", "english": True},
    {"id": "E05", "type": "英文手册", "q": "In the robot vacuum manual, what modes does the virtual wall barrier have?", "manual": "Manual32", "english": True},
    {"id": "E06", "type": "英文手册", "q": "In the water pump manual, how should I prime the pump before operation?", "manual": "Manual31", "english": True},
    {"id": "E07", "type": "英文手册", "q": "In the pressure cooker air fryer manual, what should I do before first use?", "manual": "Manual30", "english": True},
    {"id": "E08", "type": "英文手册", "q": "In the grill manual, how do I clean the grease tray?", "manual": "Manual19", "english": True},
    {"id": "E09", "type": "英文手册", "q": "In the snowmobile manual, how should I turn safely?", "manual": "Manual34", "english": True},
    {"id": "E10", "type": "英文手册", "q": "In the lawn mower manual, what should I check before mowing?", "manual": "Manual23", "english": True},
    {"id": "I01", "type": "图片", "q": "请看这张图片，能识别到什么？", "images": "red", "routes": {"image_understanding"}},
    {"id": "I02", "type": "图文手册", "q": "这张图是红色矩形；另外空调遥控器怎么换电池？", "images": "red", "manual": "Manual01"},
]


def evaluate(case: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data") or {}
    route = data.get("route") or {}
    answer = str(data.get("answer") or "")
    images = data.get("images") or []
    manual_id = str(route.get("manual_id") or "")
    route_type = str(route.get("route_type") or "")
    issues: list[str] = []

    if body.get("code") != 0:
        issues.append(f"code={body.get('code')}")
    if not answer.strip():
        issues.append("empty_answer")
    if re.search(r"(?m)^\s*#{1,6}\s*\S", answer):
        issues.append("markdown_heading")
    if answer.count("<PIC>") != len(images):
        issues.append(f"pic_image_mismatch:{answer.count('<PIC>')}!={len(images)}")
    if case.get("routes") and route_type not in case["routes"]:
        issues.append(f"route={route_type}, expected={sorted(case['routes'])}")
    if case.get("manual") and case["manual"] not in manual_id:
        issues.append(f"manual={manual_id}, expected={case['manual']}")
    if case.get("manual_contains"):
        missing = sorted(m for m in case["manual_contains"] if m not in manual_id)
        if missing:
            issues.append(f"manual_missing={missing}, got={manual_id}")
    if case.get("english"):
        cjk_count = sum(1 for ch in answer if "\u3400" <= ch <= "\u9fff")
        if cjk_count > 20:
            issues.append(f"english_answer_has_cjk:{cjk_count}")
    if case["type"] == "客服" and images:
        issues.append(f"service_has_images:{images}")

    severity = "PASS"
    if any(x.startswith(("code=", "empty", "pic_image_mismatch", "manual=", "manual_missing")) for x in issues):
        severity = "FAIL"
    elif issues:
        severity = "WARN"

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["q"],
        "status": severity,
        "issues": issues,
        "route": route_type,
        "manual_id": manual_id,
        "pic_count": answer.count("<PIC>"),
        "image_count": len(images),
        "images": images,
        "elapsed_ms": body.get("_elapsed_ms", data.get("elapsed_ms")),
        "answer_preview": answer[:220].replace("\n", " "),
    }


def main() -> None:
    token = read_token()
    red = red_png_data_url()
    results = []
    for idx, case in enumerate(CASES, 1):
        images = [red] if case.get("images") == "red" else []
        sid = f"final-smoke-{case['id']}-{int(time.time())}"
        try:
            body = call_chat(token, case["q"], session_id=sid, images=images)
            result = evaluate(case, body)
        except Exception as exc:  # noqa: BLE001
            result = {
                "id": case["id"],
                "type": case["type"],
                "question": case["q"],
                "status": "FAIL",
                "issues": [f"exception:{type(exc).__name__}:{exc}"],
                "route": "",
                "manual_id": "",
                "pic_count": 0,
                "image_count": 0,
                "images": [],
                "elapsed_ms": None,
                "answer_preview": "",
            }
        results.append(result)
        print(f"{idx:02d}/{len(CASES)} {result['id']} {result['status']} route={result['route']} manual={result['manual_id']} pics={result['pic_count']}/{result['image_count']} issues={result['issues']}")

    # Multi-turn check.
    sid = f"final-smoke-multiturn-{int(time.time())}"
    first = call_chat(token, "空调遥控器没电了，怎么更换电池？", session_id=sid)
    second = call_chat(token, "那电池型号是什么？", session_id=sid)
    mt_answer = str((second.get("data") or {}).get("answer") or "")
    mt_route = (second.get("data") or {}).get("route") or {}
    mt_issues = []
    if second.get("code") != 0:
        mt_issues.append(f"code={second.get('code')}")
    if "Manual01" not in str(mt_route.get("manual_id") or ""):
        mt_issues.append(f"manual={mt_route.get('manual_id')}")
    if "7" not in mt_answer and "七" not in mt_answer:
        mt_issues.append("battery_model_not_found")
    mt_status = "PASS" if not mt_issues else "FAIL"
    results.append(
        {
            "id": "M01",
            "type": "多轮",
            "question": "空调遥控器换电池 -> 那电池型号是什么？",
            "status": mt_status,
            "issues": mt_issues,
            "route": str(mt_route.get("route_type") or ""),
            "manual_id": str(mt_route.get("manual_id") or ""),
            "pic_count": mt_answer.count("<PIC>"),
            "image_count": len((second.get("data") or {}).get("images") or []),
            "images": (second.get("data") or {}).get("images") or [],
            "elapsed_ms": second.get("_elapsed_ms"),
            "answer_preview": mt_answer[:220].replace("\n", " "),
        }
    )
    print(f"MT M01 {mt_status} route={mt_route.get('route_type')} manual={mt_route.get('manual_id')} issues={mt_issues}")

    summary = {
        "total": len(results),
        "pass": sum(1 for r in results if r["status"] == "PASS"),
        "warn": sum(1 for r in results if r["status"] == "WARN"),
        "fail": sum(1 for r in results if r["status"] == "FAIL"),
        "by_type": {},
        "major_issues": [r for r in results if r["status"] == "FAIL"],
        "warnings": [r for r in results if r["status"] == "WARN"],
        "results": results,
    }
    for r in results:
        item = summary["by_type"].setdefault(r["type"], {"total": 0, "pass": 0, "warn": 0, "fail": 0})
        item["total"] += 1
        item[r["status"].lower()] += 1

    out_dir = Path("/root/customer_agent_deploy/app/outputs/rag_agent")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"final_user_smoke_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY_JSON=", out_path)
    print(json.dumps({k: summary[k] for k in ("total", "pass", "warn", "fail", "by_type")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
