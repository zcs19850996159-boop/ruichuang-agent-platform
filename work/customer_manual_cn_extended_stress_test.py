from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


API_URL = os.environ.get("KAFU_TEST_API_URL", "http://127.0.0.1:6006/chat")
MAX_WORKERS = int(os.environ.get("KAFU_TEST_WORKERS", "8"))


def load_token() -> str:
    candidates = [
        os.environ.get("KAFU_API_TOKEN", ""),
        "/root/customer_agent_deploy/app/.env",
        "/root/customer_agent_deploy/app/outputs/api/kafu_api_token.txt",
        "outputs/api/kafu_api_token.txt",
    ]
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if "KAFU_API_TOKEN" in text:
            for line in text.splitlines():
                if line.strip().startswith("KAFU_API_TOKEN"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        if text:
            return text
    raise RuntimeError("KAFU_API_TOKEN not found")


TOKEN = load_token()


def case(
    cid: str,
    cat: str,
    question: str,
    manuals: list[str],
    must_any: list[str],
    *,
    must_all: list[str] | None = None,
    min_images: int = 0,
    min_len: int = 45,
    multi: bool = False,
    same_manual_multi: bool = False,
) -> dict[str, Any]:
    return {
        "id": cid,
        "cat": cat,
        "question": question,
        "manuals": manuals,
        "must_any": must_any,
        "must_all": must_all or [],
        "min_images": min_images,
        "min_len": min_len,
        "multi": multi,
        "same_manual_multi": same_manual_multi,
    }


TESTS: list[dict[str, Any]] = [
    case("cn_manual_001", "Manual01-空调", "空调遥控器没电了，怎么更换电池？", ["Manual01"], ["遥控器", "电池", "正负极"], min_images=1),
    case("cn_manual_002", "Manual01-空调", "如何使用空调的自清洁运行功能？如何安装空调遥控器支架？", ["Manual01"], ["自清洁", "遥控器支架", "螺丝"], must_all=["自清洁", "支架"], min_images=2, min_len=100, same_manual_multi=True),
    case("cn_manual_003", "Manual01-空调", "空调室内机、室外机和无线遥控器分别有哪些部件？", ["Manual01"], ["室内机", "室外机", "无线遥控器"], min_images=2),
    case("cn_manual_004", "Manual01-空调", "空调手册里有哪些节能小贴士？", ["Manual01"], ["节能", "温度", "门窗"], min_len=60),
    case("cn_manual_005", "Manual02-人体工学椅", "人体工学椅怎么组装？主要步骤和部件是什么？", ["Manual02"], ["椅", "底座", "靠背"], min_images=4, min_len=90),
    case("cn_manual_006", "Manual02-人体工学椅", "人体工学椅安装时，底座、扶手和靠背应该按什么顺序装？", ["Manual02"], ["底座", "扶手", "靠背"], min_images=3),
    case("cn_manual_007", "Manual03-空气净化器", "空气净化器第一次使用前怎么拆除滤网塑料包装？", ["Manual03"], ["滤网", "塑料包装", "后盖"], min_images=3),
    case("cn_manual_008", "Manual03-空气净化器", "空气净化器滤网应该怎么清洁或更换？", ["Manual03"], ["滤网", "清洁", "更换"]),
    case("cn_manual_009", "Manual03-空气净化器", "空气净化器摆放时，距离墙壁和障碍物要注意什么？", ["Manual03"], ["3英尺", "1米", "电视", "电子设备", "阳光直射"]),
    case("cn_manual_010", "Manual04-吹风机", "使用吹风机需要佩戴哪些个人防护装备？", ["Manual04"], ["防护", "眼", "耳"], min_images=1),
    case("cn_manual_011", "Manual04-吹风机", "吹风机燃油安全有哪些注意事项？", ["Manual04"], ["燃油", "加油", "火"], min_len=70),
    case("cn_manual_012", "Manual04-吹风机", "吹风机火花塞怎么检查？", ["Manual04"], ["火花塞", "检查", "清洁"]),
    case("cn_manual_013", "Manual05-蒸汽清洁机", "蒸汽清洁机清洁硬质地面时怎么操作？", ["Manual05"], ["硬质地面", "蒸汽", "清洁"]),
    case("cn_manual_014", "Manual05-蒸汽清洁机", "蒸汽清洁机配件怎么安装和拆卸？", ["Manual05"], ["配件", "安装", "拆卸"]),
    case("cn_manual_015", "Manual05-蒸汽清洁机", "蒸汽清洁机用完以后应该怎么收纳？", ["Manual05"], ["收纳", "电源", "冷却"]),
    case("cn_manual_016", "Manual06-洗碗机", "洗碗机上喷淋臂怎么清洁？", ["Manual06"], ["上喷淋臂", "清洁", "冲洗"], min_images=1),
    case("cn_manual_017", "Manual06-洗碗机", "洗碗机下碗篮的折叠架丝怎么放下？", ["Manual06"], ["下碗篮", "折叠", "架丝"], min_images=1),
    case("cn_manual_018", "Manual06-洗碗机", "洗碗机进水管滤网如何检查和清洁？", ["Manual06"], ["进水管", "滤网", "清洁"]),
    case("cn_manual_019", "Manual11-电钻", "DCB107或DCB112充电器指示灯闪烁分别代表什么？", ["Manual11"], ["充电", "已充满", "过热", "过冷"], min_images=3),
    case("cn_manual_020", "Manual11-电钻", "电钻怎么安装钻头或附件？", ["Manual11"], ["钻头", "夹头", "安装"]),
    case("cn_manual_021", "Manual11-电钻", "电钻有哪些基本安全警告？", ["Manual11"], ["安全", "警告", "电动工具"], min_len=80),
    case("cn_manual_022", "Manual14-健身单车", "健身单车控制台有哪些功能？", ["Manual14"], ["控制台", "速度", "距离", "卡路里"], min_len=70),
    case("cn_manual_023", "Manual14-健身单车", "健身单车如何调整座椅？", ["Manual14"], ["座椅", "调整", "旋钮"], min_images=1),
    case("cn_manual_024", "Manual14-健身单车", "健身单车如何使用手握心率传感器？", ["Manual14"], ["心率", "手握", "传感器"]),
    case("cn_manual_025", "Manual16-健身追踪器", "健身追踪器表带尺寸有哪些？", ["Manual16"], ["表带", "尺寸", "环境条件"], min_images=1),
    case("cn_manual_026", "Manual16-健身追踪器", "健身追踪器如何充电？", ["Manual16"], ["充电", "触点", "USB"], min_images=1),
    case("cn_manual_027", "Manual16-健身追踪器", "健身追踪器如何佩戴和更换表带？", ["Manual16"], ["佩戴", "表带", "更换"], same_manual_multi=True),
    case("cn_manual_028", "Manual17-冰箱", "冰箱安装位置有哪些注意事项？", ["Manual17"], ["安装", "位置", "通风"], min_len=60),
    case("cn_manual_029", "Manual17-冰箱", "冰箱门打开或儿童接触时，手册提醒哪些安全事项？", ["Manual17"], ["儿童", "冰箱门", "安全"]),
    case("cn_manual_030", "Manual17-冰箱", "冰箱为什么不能放置重物或盛水容器？", ["Manual17"], ["重物", "盛水", "危险"]),
    case("cn_manual_031", "Manual18-发电机", "发电机启动前需要检查哪些内容？", ["Manual18"], ["启动前", "检查", "燃油", "机油"], min_images=1, min_len=80),
    case("cn_manual_032", "Manual18-发电机", "发电机如何连接电池或进行电池维护？", ["Manual18"], ["电池", "连接", "端子"]),
    case("cn_manual_033", "Manual18-发电机", "发电机长期存放前要怎么处理？", ["Manual18"], ["长期存放", "燃油", "机油"]),
    case("cn_manual_034", "Manual21-功能键盘", "功能键盘如何连接设备或切换模式？", ["Manual21"], ["键盘", "连接", "模式"]),
    case("cn_manual_035", "Manual21-功能键盘", "功能键盘的功能键有哪些？", ["Manual21"], ["功能键", "按键", "快捷"]),
    case("cn_manual_036", "Manual21-功能键盘", "功能键盘如何更换电池？", ["Manual21"], ["电池", "更换", "电池仓"]),
    case("cn_manual_037", "Manual26-儿童电动摩托车", "儿童电动摩托车如何给电池充电？", ["Manual26"], ["电池", "充电", "插头"]),
    case("cn_manual_038", "Manual26-儿童电动摩托车", "儿童电动摩托车使用前家长要检查什么？", ["Manual26"], ["家长", "检查", "儿童"]),
    case("cn_manual_039", "Manual26-儿童电动摩托车", "儿童电动摩托车长期不用时电池怎么保养？", ["Manual26"], ["长期", "电池", "充电"]),
    case("cn_manual_040", "Manual27-蓝牙激光鼠标", "蓝牙激光鼠标如何配对连接？", ["Manual27"], ["蓝牙", "配对", "连接"]),
    case("cn_manual_041", "Manual27-蓝牙激光鼠标", "蓝牙激光鼠标如何安装电池？", ["Manual27"], ["电池", "安装", "鼠标"]),
    case("cn_manual_042", "Manual27-蓝牙激光鼠标", "蓝牙激光鼠标DPI或按键功能怎么用？", ["Manual27"], ["DPI", "按键", "鼠标"]),
    case("cn_manual_043", "Manual28-烤箱", "烤箱顶部加热元件怎么移动？", ["Manual28"], ["顶部", "加热元件", "移动"], min_images=1),
    case("cn_manual_044", "Manual28-烤箱", "烤箱如何清洁烤盘或处理附件？", ["Manual28"], ["清洁", "烤盘", "附件"]),
    case("cn_manual_045", "Manual28-烤箱", "烤箱首次使用前要注意什么？", ["Manual28"], ["首次", "使用前", "清洁"]),
    case("cn_manual_046", "Manual29-相机", "相机如何选择AF模式？", ["Manual29"], ["AF", "模式", "对焦"]),
    case("cn_manual_047", "Manual29-相机", "相机如何安装电池或存储卡？", ["Manual29"], ["电池", "存储卡", "安装"]),
    case("cn_manual_048", "Manual29-相机", "相机如何设置日期时间？", ["Manual29"], ["日期", "时间", "设置"]),
    case("cn_manual_049", "Manual31-水泵", "水泵启动前如何灌泵或排气？", ["Manual31"], ["灌泵", "排气", "启动"]),
    case("cn_manual_050", "Manual31-水泵", "水泵运行前需要检查哪些阀门？", ["Manual31"], ["阀门", "检查", "运行"]),
    case("cn_manual_051", "Manual36-可编程温控器", "可编程温控器如何设置时间和日期？", ["Manual36"], ["时间", "日期", "设置"]),
    case("cn_manual_052", "Manual36-可编程温控器", "可编程温控器如何切换加热和制冷模式？", ["Manual36"], ["加热", "制冷", "模式"]),
    case("cn_manual_053", "Manual36-可编程温控器", "可编程温控器如何从墙上拆下？", ["Manual36"], ["墙", "拆下", "温控器"]),
    case("cn_manual_054", "Manual38-VR头显", "VR头显遮光罩怎么清洁？", ["Manual38"], ["遮光罩", "清洁", "VR"]),
    case("cn_manual_055", "Manual38-VR头显", "VR头显如何调节佩戴？", ["Manual38"], ["佩戴", "调节", "头显"]),
    case("cn_manual_056", "Manual38-VR头显", "VR头显镜片如何清洁？", ["Manual38"], ["镜片", "清洁", "头显"]),
    case("cn_manual_057", "Manual40-摩托艇", "摩托艇如何安全转弯？", ["Manual40"], ["转弯", "安全", "速度"]),
    case("cn_manual_058", "Manual40-摩托艇", "摩托艇驾驶前需要检查什么？", ["Manual40"], ["驾驶前", "检查", "安全"]),
    case("cn_manual_059", "Manual40-摩托艇", "摩托艇避免碰撞有哪些注意事项？", ["Manual40"], ["碰撞", "注意", "安全"]),
    case("cn_manual_060", "同手册多问", "健身追踪器怎么充电？表带尺寸又有哪些？", ["Manual16"], ["充电", "表带", "尺寸"], must_all=["充电", "表带"], min_images=2, min_len=100, same_manual_multi=True),
    case("cn_manual_061", "同手册多问", "发电机启动前检查什么？长期存放前又要做什么？", ["Manual18"], ["启动前", "长期存放", "燃油"], must_all=["启动", "存放"], min_len=110, same_manual_multi=True),
    case("cn_manual_062", "同手册多问", "空气净化器第一次使用怎么拆滤网包装？摆放距离有什么要求？", ["Manual03"], ["滤网", "塑料包装", "距离"], must_all=["滤网", "距离"], min_images=3, min_len=100, same_manual_multi=True),
    case("cn_manual_063", "跨手册多问", "空调遥控器怎么换电池？健身追踪器表带尺寸怎么选？", ["Manual01", "Manual16"], ["空调", "遥控器", "健身追踪器", "表带"], min_images=2, min_len=120, multi=True),
    case("cn_manual_064", "跨手册多问", "空气净化器如何拆滤网包装？洗碗机上喷淋臂怎么清洁？", ["Manual03", "Manual06"], ["空气净化器", "滤网", "洗碗机", "喷淋臂"], min_images=3, min_len=120, multi=True),
    case("cn_manual_065", "跨手册多问", "电钻充电器指示灯含义是什么？发电机启动前检查什么？", ["Manual11", "Manual18"], ["电钻", "充电器", "发电机", "启动前"], min_images=3, min_len=120, multi=True),
    case("cn_manual_066", "跨手册多问", "烤箱顶部加热元件怎么移动？VR头显遮光罩怎么清洁？", ["Manual28", "Manual38"], ["烤箱", "加热元件", "VR", "遮光罩"], min_images=1, min_len=100, multi=True),
    case("cn_manual_067", "跨手册多问", "儿童电动摩托车怎么给电池充电？蓝牙激光鼠标怎么配对？", ["Manual26", "Manual27"], ["儿童电动摩托车", "充电", "蓝牙", "配对"], min_len=110, multi=True),
    case("cn_manual_068", "易误判-客服词", "空气净化器手册里说设备有异味或异常气味时应该怎么处理？", ["Manual03"], ["异味", "空气净化器", "处理"], min_len=50),
    case("cn_manual_069", "易误判-客服词", "洗碗机手册里安装后如果需要调整高度或位置，有没有相关安装说明？", ["Manual06"], ["洗碗机", "安装", "调整"], min_len=50),
    case("cn_manual_070", "易误判-客服词", "冰箱手册里关于质量安全警告、儿童安全和放置物品有什么提醒？", ["Manual17"], ["冰箱", "安全", "儿童"], min_len=70),
]


BAD_SERVICE_PHRASES = [
    "订单号",
    "退款",
    "原支付",
    "物流",
    "快递",
    "售后规则",
    "平台介入",
]
BAD_GENERAL_PHRASES = [
    "当前证据不足以准确回答该问题",
    "没有找到相关手册内容",
    "无法根据提供的手册",
    "###",
]


def post_chat(test: dict[str, Any]) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "question": test["question"],
            "session_id": f"cn-manual-stress-{test['id']}-{uuid.uuid4().hex[:8]}",
            "images": [],
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "X-Request-Id": f"cn-manual-stress-{test['id']}-{int(time.time() * 1000)}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=75) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        payload = {"code": exc.code, "msg": raw, "data": {}}
    except Exception as exc:  # noqa: BLE001 - test harness should report transport failures.
        payload = {"code": -1, "msg": repr(exc), "data": {}}
    return payload, time.perf_counter() - started


def data_of(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def route_of(data: dict[str, Any]) -> dict[str, Any]:
    route = data.get("route")
    return route if isinstance(route, dict) else {}


def image_list(data: dict[str, Any]) -> list[str]:
    images = data.get("images")
    return [str(x) for x in images] if isinstance(images, list) else []


def route_manuals(manual_id: str, top_manuals: Any) -> set[str]:
    manuals: set[str] = set()
    for part in re.split(r"[+,;，、\s]+", manual_id or ""):
        if part.startswith("Manual"):
            manuals.add(part)
    if isinstance(top_manuals, list):
        for item in top_manuals:
            if isinstance(item, dict):
                mid = str(item.get("manual_id") or "")
                if mid.startswith("Manual"):
                    manuals.add(mid)
    return manuals


def eval_case(test: dict[str, Any], payload: dict[str, Any], elapsed: float) -> dict[str, Any]:
    data = data_of(payload)
    answer = str(data.get("answer") or "")
    route = route_of(data)
    route_type = str(route.get("route_type") or "")
    manual_id = str(route.get("manual_id") or "")
    images = image_list(data)
    pic_count = answer.count("<PIC>")
    expected_manuals = set(test["manuals"])
    routed_manuals = route_manuals(manual_id, route.get("top_manuals"))

    issues: list[str] = []
    warnings: list[str] = []
    if payload.get("code") != 0:
        issues.append(f"code={payload.get('code')}: {payload.get('msg')}")
    if route_type not in {"manual", "multi_manual"}:
        issues.append(f"route_type={route_type or '<empty>'}")
    if expected_manuals and not expected_manuals.issubset(routed_manuals):
        issues.append(f"manuals={sorted(routed_manuals)}, expect_contains={sorted(expected_manuals)}")
    if not answer.strip():
        issues.append("empty answer")
    if len(answer.strip()) < int(test.get("min_len") or 45):
        warnings.append(f"short answer len={len(answer.strip())}")
    if pic_count != len(images):
        issues.append(f"PIC/images mismatch {pic_count}!={len(images)}")
    if len(images) < int(test.get("min_images") or 0):
        issues.append(f"image_count={len(images)}, min={test.get('min_images')}")
    for phrase in BAD_GENERAL_PHRASES:
        if phrase in answer:
            issues.append(f"bad phrase: {phrase}")
    if route_type == "policy_service" or any(phrase in answer for phrase in BAD_SERVICE_PHRASES):
        issues.append("service-policy leakage")
    for term in test.get("must_all") or []:
        if term not in answer:
            issues.append(f"missing required term={term}")
    terms = test.get("must_any") or []
    if terms and not any(term in answer for term in terms):
        issues.append(f"missing any term={terms}")
    if (test.get("multi") or test.get("same_manual_multi")) and len(answer.strip()) < 100:
        warnings.append("multi-question answer may be too short")
    if elapsed > 18:
        warnings.append(f"slow {elapsed:.2f}s")

    status = "FAIL" if issues else ("WARN" if warnings else "PASS")
    return {
        "id": test["id"],
        "cat": test["cat"],
        "question": test["question"],
        "status": status,
        "route_type": route_type,
        "manual_id": manual_id,
        "routed_manuals": sorted(routed_manuals),
        "expected_manuals": sorted(expected_manuals),
        "elapsed": round(elapsed, 3),
        "answer_len": len(answer),
        "pic_count": pic_count,
        "image_count": len(images),
        "images": images,
        "issues": issues,
        "warnings": warnings,
        "answer_preview": re.sub(r"\s+", " ", answer)[:320],
    }


def run_one(test: dict[str, Any]) -> dict[str, Any]:
    payload, elapsed = post_chat(test)
    return eval_case(test, payload, elapsed)


def main() -> int:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, test): test["id"] for test in TESTS}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["id"])

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    by_cat: dict[str, dict[str, int]] = {}
    for result in results:
        counts[result["status"]] += 1
        by_cat.setdefault(result["cat"], {"PASS": 0, "WARN": 0, "FAIL": 0})[result["status"]] += 1
    elapsed_values = sorted(float(result["elapsed"]) for result in results)
    out = {
        "api_url": API_URL,
        "workers": MAX_WORKERS,
        "total": len(results),
        "counts": counts,
        "by_cat": by_cat,
        "total_elapsed": round(time.perf_counter() - started, 3),
        "avg_elapsed": round(sum(elapsed_values) / len(elapsed_values), 3),
        "p95_elapsed": elapsed_values[max(0, int(len(elapsed_values) * 0.95) - 1)],
        "failures": [result for result in results if result["status"] == "FAIL"],
        "warnings": [result for result in results if result["status"] == "WARN"],
        "samples": [result for result in results if result["status"] == "PASS"][:10],
        "results": results,
    }
    out_dir = Path("/root/customer_agent_deploy/app/outputs/rag_agent")
    if out_dir.exists():
        out_path = out_dir / f"customer_manual_cn_extended_stress_{int(time.time())}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["saved_to"] = str(out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
