from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


OUT = Path("outputs/rag_agent")


def add(
    rows: list[dict[str, str]],
    qid: int,
    category: str,
    question: str,
    manual_id: str,
    language: str,
    images: list[str],
    source_basis: str,
    note: str = "",
) -> None:
    rows.append(
        {
            "id": str(qid),
            "category": category,
            "question": question,
            "expected_route": "policy_service" if category == "policy_service" else "manual",
            "manual_id": manual_id,
            "language": language,
            "target_image_ids": ";".join(images),
            "source_basis": source_basis,
            "note": note,
        }
    )


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    policy_questions = [
        "我收到商品后想申请7天无理由退货，运费一般是谁承担？",
        "商品用了两天出现质量问题，可以换货还是只能维修？需要提供什么凭证？",
        "订单已经付款但还没发货，我现在取消的话退款多久能到账？",
        "发票抬头写错了，还能重新开电子发票吗？",
        "包裹签收时外包装破损，里面配件也少了，我应该怎么处理？",
        "商品过了7天但还在保修期内，出现故障还能走售后吗？",
        "我买错了尺寸，想换成更大的型号，差价和运费怎么处理？",
        "可以给我补发一份纸质说明书吗？如果没有纸质版怎么办？",
        "物流显示已签收但我没有收到货，这种情况要联系谁核实？",
        "商品页面写的是试用装，我收到后不满意还能退吗？",
        "我想投诉售后处理太慢，平台一般会怎么跟进？",
        "同一个订单里有两件商品，只退其中一件会影响发票吗？",
        "商品生产日期比较早，会不会影响保质期？可以申请换新吗？",
        "优惠券下单后又退款，优惠券还能退回账户吗？",
        "我想修改收货地址，但订单已经发货了，还能拦截改址吗？",
        "收到的型号和我下单的不一致，应该选择退货还是补发？",
        "如果是人为摔坏导致不能使用，官方售后还能帮忙检测维修吗？",
        "申请售后时需要上传哪些照片或视频作为凭证？",
        "商品还没拆封但包装盒有轻微压痕，会影响无理由退货吗？",
        "我想了解价格保护规则，刚买完就降价可以退差价吗？",
        "大件商品需要上门安装，可以帮我预约安装师傅吗？",
        "售后寄回检测后说没有故障，来回运费一般怎么处理？",
        "我购买时没有开发票，过了一段时间还能补开发票吗？",
        "订单拆成多个包裹发出，其中一个包裹丢失了怎么办？",
        "收到赠品损坏了，可以单独补发赠品吗？",
        "保修期是从下单时间算，还是从签收时间算？",
        "如果商品缺少配件但主机能用，可以只补发配件吗？",
        "退款会原路退回吗？银行卡和信用卡到账时间一样吗？",
        "我需要电子版说明书，客服可以发链接或文件吗？",
        "换货期间如果原商品涨价了，我还需要补差价吗？",
    ]
    for i, question in enumerate(policy_questions, 1001):
        add(rows, i, "policy_service", question, "none_policy", "zh-CN", [], "synthetic_policy", "客服题，无图")

    cn_specs = [
        ("Manual01", "空调室内机、室外机和无线遥控器分别包含哪些主要部件？", ["Manual01_0", "Manual01_1", "air_conditioner_01"], "A70 paraphrase"),
        ("Manual01", "空调遥控器没电了，按照手册应该怎样更换电池？", ["air_conditioner_01", "Manual01_2", "Manual01_3", "Manual01_4", "Manual01_5", "Manual01_22"], "A72 paraphrase"),
        ("Manual01", "我想把空调遥控器支架装到墙上，安装步骤是什么？", ["Manual01_4", "Manual01_5"], "A73 paraphrase"),
        ("Manual02", "组装这把人体工学椅时，包装里应有哪些主要零部件？", ["Manual02_0", "Manual02_1", "Manual02_2", "Manual02_3"], "A89 paraphrase"),
        ("Manual02", "人体工学椅扶手用久后有点松动，手册里是怎么解释和处理的？", ["Manual02_8"], "A90 paraphrase"),
        ("Manual03", "第一次使用空气净化器前，怎样把滤网的塑料包装拆掉并装回去？", ["Manual03_0", "Manual03_1", "Manual03_2", "Manual03_3", "Manual03_4", "Manual03_5"], "A104 paraphrase"),
        ("Manual03", "空气净化器的几种运行模式怎么切换，各自有什么作用？", ["Manual03_14", "Manual03_15", "Manual03_16", "Manual03_17"], "A106 paraphrase"),
        ("Manual03", "空气净化器上的室内空气质量指示灯代表什么含义？", ["Manual03_20"], "A107 paraphrase"),
        ("Manual04", "使用吹风机作业时，需要穿戴哪些个人防护装备？", ["Manual04_3"], "A64 paraphrase"),
        ("Manual04", "吹风机处于冷机状态时，应该怎样按步骤启动？", ["Manual04_24", "Manual04_25", "Manual04_27", "Manual04_28"], "A67 paraphrase"),
        ("Manual05", "蒸汽清洁机刚到手后，怎样快速把主机和附件组装起来？", ["Manual05_3", "Manual05_4", "Manual05_5"], "A86 paraphrase"),
        ("Manual05", "用蒸汽清洁机清洁硬质地面时，手册建议怎样操作？", [], "A88 paraphrase/no-image"),
        ("Manual06", "洗碗机整机由哪些部件组成？请按手册概括说明。", ["Dish_washer_08"], "A92 paraphrase"),
        ("Manual06", "首次使用洗碗机前，应该怎样添加专用盐？", ["Dish_washer_01", "Dish_washer_02", "Dish_washer_03"], "A94 paraphrase"),
        ("Manual06", "洗碗机洗涤剂应该加在哪里，添加步骤是什么？", ["Manual06_4", "Manual06_5"], "A95 paraphrase"),
        ("Manual11", "DCB101电钻充电器的指示灯闪烁时，各种状态分别表示什么？", ["drill0_08", "drill0_09", "drill0_10", "drill0_11", "drill0_12"], "A124 paraphrase"),
        ("Manual11", "电钻的单套无键夹头应该怎样安装？", ["drill0_01", "drill0_02", "drill0_03"], "A126 paraphrase"),
        ("Manual14", "健身单车控制台能显示和控制哪些功能？请结合手册介绍。", ["Manual14_21", "Manual14_22", "exercise_bikes_02"], "A115 paraphrase"),
        ("Manual14", "在健身单车上开始运动前，座椅或把手需要怎样调节才更舒适？", ["Manual14_24", "Manual14_25"], "A116 paraphrase"),
        ("Manual16", "健身追踪器开箱时，包装中应包含哪些物品？", ["Manual16_0", "fitness_trackers_01", "fitness_trackers_02", "fitness_trackers_03", "Manual16_3", "Manual16_21"], "A131 paraphrase"),
        ("Manual16", "健身追踪器提示电量低时，正确充电方法是什么？", ["Manual16_1", "Manual16_2"], "A132 paraphrase"),
        ("Manual17", "给冰箱连接电源时，手册提醒要注意哪些安全事项？", ["Manual17_0", "Manual17_1", "Manual17_2"], "A146 paraphrase"),
        ("Manual18", "发电机使用燃油时有哪些易燃和有毒方面的安全提醒？", ["generator_20", "generator_21", "generator_03"], "A153 paraphrase"),
        ("Manual18", "发电机需要打开燃油开关供油时，具体操作步骤是什么？", ["Manual18_19", "Manual18_20"], "A156 paraphrase"),
        ("Manual21", "功能键盘初次使用时，手册中的设置步骤是什么？", ["Manual21_1", "Manual21_2", "function_keyboard_01"], "A202 paraphrase"),
        ("Manual27", "蓝牙激光鼠标应该怎样安装电池？", ["Manual27_1", "Manual27_2", "Manual27_3"], "A208 paraphrase"),
        ("Manual28", "如果要拆下烤箱门，应该按哪些步骤操作？", ["oven_01", "oven_02", "oven_06", "Manual28_7"], "A216 paraphrase"),
        ("Manual31", "清洗水泵油箱滤网时，应按什么顺序拆洗和装回？", ["Manual31_31", "Manual31_32", "Manual31_33", "Manual31_34"], "A183 paraphrase"),
        ("Manual36", "可编程温控器要设置日期和时间时，应该怎么操作？", ["Manual36_25", "thermostat_07", "Manual36_26"], "A186 paraphrase"),
        ("Manual40", "在深水中重新登上摩托艇并保持平衡，手册建议怎样做？", ["Manual40_13", "Manual40_14", "Manual40_15", "Manual40_16"], "A174 paraphrase"),
    ]
    for idx, (manual_id, question, images, basis) in enumerate(cn_specs, 1031):
        add(rows, idx, "cn_manual", question, manual_id, "zh-CN", images, basis, "中文手册题")

    en_specs = [
        ("Manual07", "How do I turn on and use the coffee machine energy saving mode?", ["Manual07_4", "Manual07_5"], "A265 paraphrase"),
        ("Manual07", "How can I change the default energy-saving setting on the coffee machine?", ["Manual07_6", "Manual07_7", "Manual07_8", "Manual07_9"], "A266 paraphrase"),
        ("Manual07", "What steps should I follow to program the coffee machine water volume?", ["Manual07_24", "Manual07_25", "Manual07_26", "Manual07_27"], "A267 paraphrase"),
        ("Manual07", "Before storage, frost protection, or maintenance, how should I empty the coffee machine system?", ["Manual07_28", "Manual07_29", "Manual07_30", "Manual07_31", "Manual07_32"], "A268 paraphrase"),
        ("Manual08", "Before using the air fryer for the first time, what preparation does the manual require?", ["Manual08_5"], "A241 paraphrase"),
        ("Manual09", "Where can I find the boat emission control certificate approval label?", ["Manual09_6", "Manual09_7"], "A242 paraphrase"),
        ("Manual09", "Before sailing, how should I operate the boat battery conversion switches?", ["Manual09_42", "Manual09_43", "Manual09_44"], "A243 paraphrase"),
        ("Manual09", "Can you explain how the boat steering system works according to the manual?", [], "A244 paraphrase/no-image"),
        ("Manual09", "What should I do when the boat shows an over-temperature warning?", ["Manual09_95"], "A245 paraphrase"),
        ("Manual10", "When the camera battery is not installed in the camera, how should I recharge it?", ["Manual10_12", "Manual10_13", "Manual10_14", "Manual10_15"], "A280 paraphrase"),
        ("Manual10", "How do I install the camera battery after making sure the camera is powered off?", ["Manual10_16", "Camera_09", "Camera_10", "Camera_12", "Camera_13"], "A281 paraphrase"),
        ("Manual10", "How can I power the camera from a household electrical outlet?", ["Camera_14", "Camera_15", "Manual10_19", "Manual10_20"], "A282 paraphrase"),
        ("Manual10", "What is the correct way to mount a lens on the camera before shooting?", ["Manual10_21", "Manual10_22", "Manual10_23", "Manual10_24"], "A283 paraphrase"),
        ("Manual12", "What components should be included when I receive the earphones?", ["Manual12_0", "Manual12_1", "Manual12_2", "Manual12_3", "Manual12_4"], "A296 paraphrase"),
        ("Manual12", "If the charging case battery is low, how do I charge the earphones case?", ["earphones_04", "earphones_05", "earphones_06", "earphones_07"], "A297 paraphrase"),
        ("Manual12", "During Bluetooth pairing and connection, what status scenarios might the earphones show?", ["Manual12_5", "Manual12_6", "Manual12_7"], "A298 paraphrase"),
        ("Manual12", "What are the main touch or button control functions of the earphones?", ["Manual12_8", "Manual12_9"], "A299 paraphrase"),
        ("Manual13", "From the different views of the eReader, what buttons and interfaces are shown?", ["Manual13_0", "Manual13_1"], "A303 paraphrase"),
        ("Manual13", "What do the Main Menu and Browser History functions do on the eReader?", ["Manual13_4"], "A304 paraphrase"),
        ("Manual13", "In eBook mode, what happens after I press the M button on the eReader?", ["eReader_08", "Manual13_5", "Manual13_6"], "A305 paraphrase"),
        ("Manual13", "If I want to listen to music on the eReader, what should I select or operate?", ["Manual13_7", "Manual13_8"], "A306 paraphrase"),
        ("Manual15", "When connecting the fax machine, what procedure should I follow to complete the setup?", ["Manual15_2"], "A311 paraphrase"),
        ("Manual15", "What safety precautions should I follow when using the fax machine?", ["Manual15_6", "Manual15_7", "Manual15_8", "Manual15_9"], "A312 paraphrase"),
        ("Manual15", "How should I keep my fingers safe around the fax machine mechanisms?", ["Manual15_6", "Manual15_7", "Manual15_8", "Manual15_9", "fax_08", "Manual15_10", "Manual15_11", "Manual15_12"], "A313 paraphrase"),
        ("Manual15", "Before moving the fax machine, what should I pay attention to?", ["fax_08"], "A314 paraphrase"),
        ("Manual19", "How do I connect the grill regulator to the LP tank?", ["Manual19_16", "Manual19_17", "Manual19_18"], "A317 paraphrase"),
        ("Manual19", "For grill safety, how should I perform leak testing on valves, hose, and regulator?", ["Manual19_20", "Manual19_21"], "A318 paraphrase"),
        ("Manual19", "What should I know about indirect cooking when using the grill?", ["Manual19_36"], "A319 paraphrase"),
        ("Manual19", "What are the first steps in the grill assembly process?", ["Manual19_49", "Manual19_50", "Manual19_51", "Manual19_52"], "A320 paraphrase"),
        ("Manual20", "Where are the identification numbers located on my Jet Ski or watercraft?", ["Manual20_0", "jetski_01"], "A322 paraphrase"),
        ("Manual20", "What cruising limitations does the watercraft manual describe?", ["Manual20_16", "Manual20_17"], "A323 paraphrase"),
        ("Manual20", "What operating requirements should I satisfy before using the Jet Ski?", ["Manual20_15", "Manual20_19"], "A325 paraphrase"),
        ("Manual22", "Can you give me an overview of the landline base station parts?", ["Manual22_18"], "A351 paraphrase"),
        ("Manual22", "How should I connect the base station for the landline phone?", ["Manual22_21"], "A352 paraphrase"),
        ("Manual22", "What is the correct way to install the landline handset?", ["Manual22_23", "Manual22_25"], "A353 paraphrase"),
        ("Manual22", "How can I check the battery level on the landline phone?", ["Manual22_28", "Manual22_29"], "A354 paraphrase"),
        ("Manual23", "How do I lower the roll bar on the lawn mower?", ["Manual23_32"], "A358 paraphrase"),
        ("Manual23", "How can I adjust the rear-shock assemblies on the suspension lawn mower?", ["Manual23_37", "Manual23_38"], "A359 paraphrase"),
        ("Manual24", "How do I set the light timer on the over-the-range microwave?", ["Manual24_11", "Manual24_12"], "A367 paraphrase"),
        ("Manual32", "What is the proper procedure for emptying the robot vacuum cleaner bin?", ["Manual32_8", "Manual32_9"], "A403 paraphrase"),
    ]
    for idx, (manual_id, question, images, basis) in enumerate(en_specs, 1061):
        add(rows, idx, "en_manual", question, manual_id, "en-US", images, basis, "英文手册题")

    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "category",
        "question",
        "expected_route",
        "manual_id",
        "language",
        "target_image_ids",
        "source_basis",
        "note",
    ]
    labeled_path = OUT / "synthetic_100_new_questions_labeled.csv"
    questions_path = OUT / "synthetic_100_new_questions.csv"
    teacher_path = OUT / "synthetic_100_new_questions_teacher.jsonl"
    json_path = OUT / "synthetic_100_new_questions_labeled.json"
    summary_path = OUT / "synthetic_100_new_questions_summary.json"

    with labeled_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with questions_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "question": row["question"]})
    with teacher_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "manual_id": row["manual_id"],
                        "route_type": row["expected_route"],
                        "category": row["category"],
                        "image_ids": [image_id for image_id in row["target_image_ids"].split(";") if image_id],
                        "source_basis": row["source_basis"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(row["category"] for row in rows)
    manual_counts = Counter(row["manual_id"] for row in rows if row["manual_id"] != "none_policy")
    summary = {
        "rows": len(rows),
        "category_counts": dict(counts),
        "manual_count": len(manual_counts),
        "manual_counts": dict(sorted(manual_counts.items())),
        "files": {
            "questions_csv": str(questions_path),
            "labeled_csv": str(labeled_path),
            "teacher_jsonl": str(teacher_path),
            "labeled_json": str(json_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    rows = build_rows()
    counts = Counter(row["category"] for row in rows)
    assert len(rows) == 100, len(rows)
    assert counts == {"policy_service": 30, "cn_manual": 30, "en_manual": 40}, counts
    write_outputs(rows)


if __name__ == "__main__":
    main()
