from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


GROUPS: dict[str, list[list[str]]] = {
    "blind-img-013": [["预热", "heating"], ["25"], ["常亮", "steady", "准备好"]],
    "blind-img-014": [["Magic eye"], ["关闭", "off"]],
    "blind-img-015": [["On/off"], ["开始", "start"]],
    "blind-img-016": [["Smart Programs"], ["Start", "启动"], ["Cancel", "取消"]],
    "blind-img-017": [["功能"], ["温度"], ["归零", "0"]],
    "blind-img-018": [["通气旋钮"], ["ON"], ["OFF"], ["燃油"]],
    "blind-img-019": [["冲击钻"], ["调节环", "C"], ["砌体", "硬质合金"]],
    "blind-img-020": [["电池"], ["充电"], ["信号", "基座"]],
    "blind-img-021": [["T型轨", "T 型轨", "T-rail"], ["2毫米", "2 mm", "5/64"]],
    "blind-img-022": [["Pilot screw", "混合比", "怠速"], ["稀", "leaner"], ["暖", "高海拔"], ["浓", "richer"]],
    "blind-img-023": [["Main jet", "主喷油嘴"], ["较小", "Smaller"], ["稀", "leaner"], ["较大", "Larger"], ["浓", "richer"]],
    "blind-img-024": [["60厘米", "60 cm", "2英尺", "2 ft"], ["不得", "不要", "切勿"]],
    "blind-img-025": [["左"], ["右"], ["快门"], ["模式"]],
    "blind-img-026": [["High"], ["Natural Wind"], ["风速"]],
    "blind-img-027": [["BPR4ES"], ["0.7", "0.8"]],
    "blind-img-028": [["充电触点", "charging contacts"], ["干布", "dry cloth"]],
    "blind-img-029": [["两根", "2根", "two"], ["Rc"], ["向下", "down"]],
    "blind-img-030": [["VR"], ["追踪灯", "麦克风", "遮光罩"]],
    "blind-img-031": [["延长管"], ["喷射喷嘴"], ["角形喷嘴"]],
    "blind-img-032": [["第一个环扣", "first loop"]],
    "blind-img-033": [["充电线", "charging cable"], ["USB"]],
    "blind-img-034": [["侧燃烧器", "side burner"], ["蝶形螺母", "wing nut"], ["阀门", "valve"]],
    "blind-img-035": [["UWP"], ["1/4"], ["电线"]],
    "blind-img-036": [["安全盖", "safety cover"], ["内盖", "inner lid"], ["关闭", "close"]],
    "blind-img-037": [["10–21", "10-21"], ["1–4", "1-4"], ["1/2–2", "1/2-2"]],
    "blind-img-038": [["OFF"], ["断开", "disconnect"], ["灯泡"]],
    "blind-img-039": [["上盖"], ["向上"], ["传感器"]],
    "blind-img-040": [["燃烧器"], ["电极"], ["火箱"]],
    "blind-img-041": [["前灯"], ["线束", "插头"]],
    "blind-img-042": [["盐箱"], ["逆时针"], ["专用盐"]],
    "blind-img-043": [["肩带"], ["吹管", "吹风机"]],
    "blind-img-044": [["爆炸图", "零件"], ["成人"]],
    "blind-img-045": [["Halo"], ["24英寸", "24 英寸", "60厘米", "60 cm"], ["保护"]],
    "blind-img-046": [["油门扳机"], ["油门锁"]],
    "blind-img-047": [["维护"], ["0.5"], ["火花塞"]],
    "blind-img-048": [["右侧"], ["按钮"], ["电量"]],
    "blind-img-049": [["8字", "8 字"], ["缩小"]],
    "blind-img-050": [["15–21", "15-21"], ["8字", "8 字"], ["拖曳"]],
    "blind-img-051": [["USB"], ["端口"]],
    "blind-img-052": [["禁止"]],
    "blind-img-053": [["腕托"], ["磁"]],
    "blind-img-054": [["艇尾"], ["翻转", "扶正"]],
    "blind-img-055": [["电池仓盖"], ["盖回", "装回", "复位"]],
    "blind-img-056": [["滑航"], ["椭圆"], ["8字", "8 字"]],
    "blind-img-057": [["易燃喷雾"], ["禁止", "不得"]],
    "blind-img-058": [["运动部件"], ["警告", "危险"]],
    "blind-img-059": [["USB-C"], ["充电盒"]],
    "blind-img-060": [["左耳", "L"], ["按住"], ["提示音"], ["语音助手", "音乐"]],
}


FORBIDDEN: dict[str, list[str]] = {
    "blind-img-013": ["除垢"],
    "blind-img-018": ["发动机机油"],
    "blind-img-019": ["电池组安装"],
    "blind-img-022": ["拆下放油螺塞"],
    "blind-img-026": ["安装电池"],
    "blind-img-028": ["提取器", "滚刷"],
    "blind-img-029": ["设置日期"],
    "blind-img-030": ["用水清洗"],
    "blind-img-033": ["扣紧表带"],
    "blind-img-036": ["部件总览"],
    "blind-img-039": ["空气质量指示灯"],
    "blind-img-041": ["挡泥板"],
    "blind-img-043": ["停机开关"],
    "blind-img-046": ["停机开关"],
    "blind-img-049": ["三种速度"],
    "blind-img-050": ["三种速度"],
    "blind-img-052": ["搬运"],
    "blind-img-053": ["配置文件"],
    "blind-img-054": ["三种速度"],
    "blind-img-055": ["WIDCOMM", "配对"],
    "blind-img-057": ["搬运"],
    "blind-img-058": ["搬运"],
    "blind-img-060": ["蓝牙配对"],
}


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "outputs" / "blind_media_benchmark_v1" / "manifest.jsonl"),
    )
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    manifest = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["case_id"] in GROUPS
        )
    }
    results: dict[str, dict[str, Any]] = {}
    for path in args.results:
        results.update(read_jsonl(Path(path)))

    scored = []
    for case_id, groups in GROUPS.items():
        expected = manifest[case_id]["expected"]
        result = results.get(case_id, {})
        answer = str(result.get("answer") or "")
        images = [str(value) for value in result.get("actual_images") or []]
        group_hits = [
            any(str(term).lower() in answer.lower() for term in group)
            for group in groups
        ]
        forbidden_hits = [
            term
            for term in FORBIDDEN.get(case_id, [])
            if term.lower() in answer.lower()
        ]
        row = {
            "case_id": case_id,
            "manual_match": str(result.get("actual_manual_id") or "")
            == str(expected.get("manual_id") or ""),
            "answer_groups_pass": all(group_hits),
            "answer_group_hits": group_hits,
            "forbidden_pass": not forbidden_hits,
            "forbidden_hits": forbidden_hits,
            "image_supported": str(expected.get("image_id") or "") in images,
            "actual_manual_id": result.get("actual_manual_id"),
            "actual_images": images,
            "human_approval_required": True,
        }
        row["automatic_pass"] = all(
            bool(row[key])
            for key in ("manual_match", "answer_groups_pass", "forbidden_pass", "image_supported")
        )
        scored.append(row)

    summary = {
        "total": len(scored),
        "automatic_pass": sum(bool(row["automatic_pass"]) for row in scored),
        "manual_match": sum(bool(row["manual_match"]) for row in scored),
        "answer_groups_pass": sum(bool(row["answer_groups_pass"]) for row in scored),
        "forbidden_pass": sum(bool(row["forbidden_pass"]) for row in scored),
        "image_supported": sum(bool(row["image_supported"]) for row in scored),
        "human_approval_required": True,
    }
    payload = {"summary": summary, "cases": scored}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
