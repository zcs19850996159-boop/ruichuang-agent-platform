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
    min_len: int = 55,
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
    case("en_manual_001", "Manual07-coffee machine", "For the coffee machine, how should I clean the coffee outlet and maintenance unit?", ["Manual07"], ["coffee outlet", "maintenance unit", "clean"], min_images=2),
    case("en_manual_002", "Manual07-coffee machine", "How do I descale the coffee machine according to water hardness?", ["Manual07"], ["descale", "water hardness", "descaling"]),
    case("en_manual_003", "Manual07-coffee machine", "How do I clean the milk frother or steam nozzle on the coffee machine?", ["Manual07"], ["milk frother", "steam nozzle", "clean"]),
    case("en_manual_004", "Manual08-air fryer", "What should I know before using the air fryer for the first time?", ["Manual08"], ["air fryer", "first time", "smoke"]),
    case("en_manual_005", "Manual08-air fryer", "How should I clean the air fryer pan and basket after use?", ["Manual08"], ["pan", "basket", "clean"], min_images=1),
    case("en_manual_006", "Manual08-air fryer", "Can I fill the air fryer pan with oil or frying fat?", ["Manual08"], ["oil", "frying fat", "hot air"], min_images=1),
    case("en_manual_007", "Manual09-boat", "anchor light", ["Manual09"], ["anchor light", "bow light", "switch"], min_images=3, min_len=80),
    case("en_manual_008", "Manual09-boat", "How do I check the anchor light switch on the boat?", ["Manual09"], ["anchor light", "bow light", "middle position"], min_images=4, min_len=120),
    case("en_manual_009", "Manual09-boat", "boat turning at planing speed", ["Manual09"], ["planing speed", "turn", "throttle"], min_images=1, min_len=80),
    case("en_manual_010", "Manual09-boat", "What does the boat maintenance setting screen show and how does Reset work?", ["Manual09"], ["maintenance", "running hours", "Reset"], min_images=3),
    case("en_manual_011", "Manual09-boat", "How do I turn the boat water supply on or off?", ["Manual09"], ["water supply", "shut-off valve", "inspection cover"], min_images=2),
    case("en_manual_012", "Manual09-boat", "How can I listen to music from my phone through the boat sound system?", ["Manual09"], ["stereo", "USB", "Bluetooth"], min_images=2),
    case("en_manual_013", "Manual09-boat", "What fire extinguisher does the boat need?", ["Manual09"], ["fire extinguisher", "5-B", "aboard"], min_images=2),
    case("en_manual_014", "Manual09-boat", "Before starting the boat engines, what checks and key steps are required?", ["Manual09"], ["battery switch", "blower", "lanyard", "START"], min_images=3, min_len=130),
    case("en_manual_015", "Manual10-camera", "AF Mode", ["Manual10"], ["AF mode", "One-Shot", "AI Servo", "AI Focus"], min_images=3, min_len=90),
    case("en_manual_016", "Manual10-camera", "How do I select AF Mode on the camera?", ["Manual10"], ["AF-WB", "One-Shot", "AI Servo"], min_images=3),
    case("en_manual_017", "Manual10-camera", "How does CP Direct printing work from the camera?", ["Manual10"], ["CP Direct", "printer", "direct printing"], min_images=2),
    case("en_manual_018", "Manual10-camera", "How do I focus on an off-center subject with the camera?", ["Manual10"], ["focus lock", "shutter", "recompose"], min_images=2),
    case("en_manual_019", "Manual10-camera", "How can I turn the camera beeper off?", ["Manual10"], ["Beep", "Off", "menu"]),
    case("en_manual_020", "Manual12-earphones", "How should I maintain and care for the earphones before charging?", ["Manual12"], ["earphones", "dry before charging", "soft cloth"]),
    case("en_manual_021", "Manual12-earphones", "How do I clean or replace the ear tips on the earbuds?", ["Manual12"], ["ear tip", "earbud", "clean"]),
    case("en_manual_022", "Manual12-earphones", "What are the charging case and battery specs for the earphones?", ["Manual12"], ["charging case", "battery", "play time"]),
    case("en_manual_023", "Manual13-ereader", "In eBook mode, how can I change font size and page display comfort?", ["Manual13"], ["eBook", "Zoom", "font size"], min_images=1),
    case("en_manual_024", "Manual13-ereader", "How do I connect the e-reader to a computer and manage files?", ["Manual13"], ["USB", "removable disk", "files"]),
    case("en_manual_025", "Manual13-ereader", "How do I adjust display settings or backlight on the e-reader?", ["Manual13"], ["Display Settings", "brightness", "light"], min_images=2),
    case("en_manual_026", "Manual15-fax", "fax safety", ["Manual15"], ["fax", "safety", "injury"], min_images=1),
    case("en_manual_027", "Manual15-fax", "In order to keep my fingers safe, what should I pay attention to about the fax?", ["Manual15"], ["fingers", "pages", "injury"], min_images=1),
    case("en_manual_028", "Manual15-fax", "Are caution or warning labels inside the fax important? Can I remove them?", ["Manual15"], ["caution", "warning labels", "remove"], min_images=2),
    case("en_manual_029", "Manual15-fax", "If I am in Canada, what fax compliance or country-use statement applies?", ["Manual15"], ["Canada", "Industry Canada", "USA"]),
    case("en_manual_030", "Manual19-grill", "When using the grill, how do I connect the regulator to the LP tank?", ["Manual19"], ["regulator", "LP tank", "coupling nut"], min_images=3),
    case("en_manual_031", "Manual19-grill", "How do I leak test the grill valves, hose and regulator?", ["Manual19"], ["leak", "valves", "hose", "regulator"], min_images=3),
    case("en_manual_032", "Manual19-grill", "What should I know about indirect cooking on the grill?", ["Manual19"], ["indirect cooking", "lid", "poultry"], min_images=1),
    case("en_manual_033", "Manual19-grill", "How should I clean and store the grill for long-term storage?", ["Manual19"], ["clean", "storage", "grill"]),
    case("en_manual_034", "Manual20-jet ski", "What are the operating requirements before using the jet ski?", ["Manual20"], ["operator", "PFD", "maximum load"], min_images=2),
    case("en_manual_035", "Manual20-jet ski", "How should I board the jet ski from deep water and keep balance?", ["Manual20"], ["deep water", "balance", "boarding"], min_images=1),
    case("en_manual_036", "Manual20-jet ski", "How does the jet ski turn and why should I not release the throttle while turning?", ["Manual20"], ["turn", "throttle", "steering"], min_images=1),
    case("en_manual_037", "Manual22-landline", "How do I add or use phonebook contacts on the landline?", ["Manual22"], ["phonebook", "contact", "number"]),
    case("en_manual_038", "Manual22-landline", "How can I set the ringer melody or volume on the landline?", ["Manual22"], ["ringer", "melody", "volume"]),
    case("en_manual_039", "Manual22-landline", "What does searching status mean on the landline base or handset?", ["Manual22"], ["searching", "base", "handset"]),
    case("en_manual_040", "Manual23-lawn mower", "How do I lower the roll bar on the lawn mower?", ["Manual23"], ["roll bar", "hairpin", "pins"], min_images=2),
    case("en_manual_041", "Manual23-lawn mower", "How do I adjust the height of cut on the lawn mower?", ["Manual23"], ["height of cut", "deck", "raise"], min_images=2),
    case("en_manual_042", "Manual23-lawn mower", "How do I replace the mower belt?", ["Manual23"], ["mower belt", "idler", "spring"], min_images=2),
    case("en_manual_043", "Manual23-lawn mower", "How do I remove the filters on the lawn mower?", ["Manual23"], ["filter", "remove", "mower"]),
    case("en_manual_044", "Manual24-microwave", "charcoal filter", ["Manual24"], ["charcoal filter", "replace", "microwave"], min_images=3, min_len=80),
    case("en_manual_045", "Manual24-microwave", "How do I replace the microwave charcoal filter?", ["Manual24"], ["charcoal filter", "replace", "grille"], min_images=3),
    case("en_manual_046", "Manual24-microwave", "How do I store and recall a favorite recipe on the microwave?", ["Manual24"], ["Favorite Recipe", "store", "recall"], min_images=2),
    case("en_manual_047", "Manual24-microwave", "How does microwave Auto Defrost work for ground beef?", ["Manual24"], ["Auto Defrost", "ground beef", "START"], min_images=2),
    case("en_manual_048", "Manual25-motherboard", "What does the motherboard CMOS/RTC battery preserve and when might it need replacement?", ["Manual25"], ["CMOS", "RTC", "battery"], min_images=1),
    case("en_manual_049", "Manual25-motherboard", "How do I install the CPU on the motherboard?", ["Manual25"], ["CPU", "socket", "load plate"], min_images=2),
    case("en_manual_050", "Manual25-motherboard", "How do I create RAID in BIOS on the motherboard?", ["Manual25"], ["RAID", "BIOS", "SATA"], min_images=2),
    case("en_manual_051", "Manual25-motherboard", "Where are the TPM and serial port connectors on the motherboard?", ["Manual25"], ["TPM", "serial port", "connector"], min_images=1),
    case("en_manual_052", "Manual30-pressure cooker", "How does Natural Release (NR or NPR) work on the multi-use pressure cooker?", ["Manual30"], ["Natural Release", "float valve", "pressure"], min_images=1),
    case("en_manual_053", "Manual30-pressure cooker", "How do I remove and close the pressure cooking lid?", ["Manual30"], ["lid", "counterclockwise", "clockwise"], min_images=2),
    case("en_manual_054", "Manual30-pressure cooker", "How do I use Delay Start on the pressure cooker?", ["Manual30"], ["Delay Start", "timer", "pressure"]),
    case("en_manual_055", "Manual32-robot vacuum", "virtual wall barrier", ["Manual32"], ["Virtual Wall", "Halo", "barrier"], min_images=2, min_len=80),
    case("en_manual_056", "Manual32-robot vacuum", "What are the two primary modes of the robot vacuum?", ["Manual32"], ["CLEAN", "SPOT", "recharge"]),
    case("en_manual_057", "Manual32-robot vacuum", "Describe the robot vacuum anatomy and main parts.", ["Manual32"], ["sensors", "buttons", "wheels", "brush", "charging contacts"], min_images=1),
    case("en_manual_058", "Manual32-robot vacuum", "How do I set an automatic cleaning schedule for the robot vacuum?", ["Manual32"], ["schedule", "CLEAN", "time"]),
    case("en_manual_059", "Manual33-security camera", "How do I power the security camera?", ["Manual33"], ["PoE", "Ethernet", "802.3af"], min_len=70),
    case("en_manual_060", "Manual33-security camera", "How do I wall mount and aim the security camera?", ["Manual33"], ["wall mount", "mount plate", "Dashboard"], min_images=3),
    case("en_manual_061", "Manual34-snowmobile", "How should I ride a snowmobile uphill?", ["Manual34"], ["uphill", "lean", "throttle"], min_images=2),
    case("en_manual_062", "Manual34-snowmobile", "How do I inspect the snowmobile spark plug?", ["Manual34"], ["spark plug", "electrode gap", "insulator"], min_images=3),
    case("en_manual_063", "Manual34-snowmobile", "How do I adjust the snowmobile throttle cable?", ["Manual34"], ["throttle cable", "free play", "locknut"], min_images=4),
    case("en_manual_064", "Manual35-television", "What should I do about poor reception, ghosts or snow on the television?", ["Manual35"], ["poor reception", "ghosts", "snow"], min_images=2),
    case("en_manual_065", "Manual35-television", "How do I use caption or on-screen text settings on the TV?", ["Manual35"], ["CAPTION", "Mode1", "Text1"], min_images=1),
    case("en_manual_066", "Manual35-television", "How does the TV sleep timer work?", ["Manual35"], ["SLEEP", "timer", "standby"], min_images=1),
    case("en_manual_067", "Manual37-toothbrush", "How do I activate or deactivate toothbrush smart features?", ["Manual37"], ["activate", "deactivate", "app"], min_images=2),
    case("en_manual_068", "Manual37-toothbrush", "How should I clean the toothbrush brush head and handle?", ["Manual37"], ["brush head", "handle", "rinse"], min_images=3),
    case("en_manual_069", "Manual37-toothbrush", "How do I charge the toothbrush with the travel case?", ["Manual37"], ["travel case", "USB", "charging"], min_images=2),
    case("en_manual_070", "Manual39-washing machine", "What are the washing machine washing, spin dry and rinsing procedures?", ["Manual39"], ["washing", "spin dry", "rinsing"], min_images=1),
    case("en_manual_071", "Manual39-washing machine", "How do I connect the drain hose and clean the washing machine?", ["Manual39"], ["drain hose", "cleaning", "machine"], min_images=1),
    case("en_manual_072", "Manual39-washing machine", "What troubleshooting checks are listed for the washing machine?", ["Manual39"], ["troubleshooting", "washing machine", "check"], min_len=60),
    case("en_manual_073", "same-manual multi", "For the camera, how do I select AF Mode and then use CP Direct printing?", ["Manual10"], ["AF mode", "CP Direct", "printer"], must_all=["AF", "CP Direct"], min_images=4, min_len=130, same_manual_multi=True),
    case("en_manual_074", "same-manual multi", "For the boat, explain the anchor light switch and the maintenance setting screen.", ["Manual09"], ["anchor light", "maintenance", "Reset"], must_all=["anchor", "maintenance"], min_images=5, min_len=150, same_manual_multi=True),
    case("en_manual_075", "same-manual multi", "For the robot vacuum, explain virtual wall barrier modes and the two primary cleaning modes.", ["Manual32"], ["Virtual Wall", "CLEAN", "SPOT"], must_all=["Virtual Wall", "CLEAN"], min_images=3, min_len=140, same_manual_multi=True),
    case("en_manual_076", "cross-manual multi", "How do I replace the microwave charcoal filter, and how do I set robot vacuum scheduling?", ["Manual24", "Manual32"], ["charcoal filter", "schedule", "robot vacuum"], min_images=3, min_len=150, multi=True),
    case("en_manual_077", "cross-manual multi", "How do I select camera AF Mode, and what should I do for fax finger safety?", ["Manual10", "Manual15"], ["AF mode", "fingers", "fax"], min_images=3, min_len=140, multi=True),
    case("en_manual_078", "cross-manual multi", "How do I use pressure cooker Natural Release, and how do I inspect a snowmobile spark plug?", ["Manual30", "Manual34"], ["Natural Release", "spark plug", "electrode"], min_images=3, min_len=150, multi=True),
]


IMAGE_PREFIX_MANUAL = {
    "Camera": "Manual10",
    "television0": "Manual35",
    "toothbrush0": "Manual37",
    "robot_vacuum": "Manual32",
}

BAD_GENERAL_PHRASES = [
    "current evidence is insufficient",
    "does not contain any information",
    "I don't have enough information",
    "model answer generation failed",
    "model service is temporarily unavailable",
    "###",
]
BAD_SERVICE_PHRASES = [
    "order number",
    "refund",
    "return policy",
    "shipping fee",
    "logistics",
    "after-sales rule",
    "platform intervention",
]


def post_chat(test: dict[str, Any]) -> tuple[dict[str, Any], float]:
    body = json.dumps(
        {
            "question": test["question"],
            "session_id": f"en-manual-stress-{test['id']}-{uuid.uuid4().hex[:8]}",
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
            "X-Request-Id": f"en-manual-stress-{test['id']}-{int(time.time() * 1000)}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        payload = {"code": exc.code, "msg": raw, "data": {}}
    except Exception as exc:  # noqa: BLE001 - test harness reports transport failures.
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


def manual_from_image(image_id: str) -> str:
    for prefix, manual_id in IMAGE_PREFIX_MANUAL.items():
        if image_id.startswith(prefix):
            return manual_id
    match = re.match(r"(Manual\d+)_", image_id)
    return match.group(1) if match else ""


def route_manuals(manual_id: str, top_manuals: Any, images: list[str]) -> set[str]:
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
    for image_id in images:
        mid = manual_from_image(image_id)
        if mid:
            manuals.add(mid)
    return manuals


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def eval_case(test: dict[str, Any], payload: dict[str, Any], elapsed: float) -> dict[str, Any]:
    data = data_of(payload)
    answer = str(data.get("answer") or "")
    answer_lower = answer.lower()
    route = route_of(data)
    route_type = str(route.get("route_type") or "")
    manual_id = str(route.get("manual_id") or "")
    images = image_list(data)
    pic_count = answer.count("<PIC>")
    expected_manuals = set(test["manuals"])
    routed_manuals = route_manuals(manual_id, route.get("top_manuals"), images)

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
    if len(answer.strip()) < int(test.get("min_len") or 55):
        warnings.append(f"short answer len={len(answer.strip())}")
    if pic_count != len(images):
        issues.append(f"PIC/images mismatch {pic_count}!={len(images)}")
    if len(images) < int(test.get("min_images") or 0):
        issues.append(f"image_count={len(images)}, min={test.get('min_images')}")
    if cjk_count(answer) > 8:
        issues.append(f"English question answered with too much Chinese cjk={cjk_count(answer)}")
    for phrase in BAD_GENERAL_PHRASES:
        if phrase in answer_lower:
            issues.append(f"bad phrase: {phrase}")
    if route_type == "policy_service" or any(phrase in answer_lower for phrase in BAD_SERVICE_PHRASES):
        issues.append("service-policy leakage")
    for term in test.get("must_all") or []:
        if term.lower() not in answer_lower:
            issues.append(f"missing required term={term}")
    terms = test.get("must_any") or []
    if terms and not any(term.lower() in answer_lower for term in terms):
        issues.append(f"missing any term={terms}")
    if (test.get("multi") or test.get("same_manual_multi")) and len(answer.strip()) < 120:
        warnings.append("multi-question answer may be too short")
    if elapsed > 22:
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
        "cjk_count": cjk_count(answer),
        "pic_count": pic_count,
        "image_count": len(images),
        "images": images,
        "issues": issues,
        "warnings": warnings,
        "answer_preview": re.sub(r"\s+", " ", answer)[:360],
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
        out_path = out_dir / f"customer_manual_en_extended_stress_{int(time.time())}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        out["saved_to"] = str(out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
