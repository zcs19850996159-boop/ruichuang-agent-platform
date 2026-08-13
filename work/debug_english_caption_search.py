import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path("outputs/rag_agent").resolve()))
from rag_core import tokenize


STOP = {
    "how", "what", "do", "does", "the", "a", "an", "i", "you", "to", "of",
    "if", "is", "are", "can", "should", "my", "this", "that", "when",
    "while", "using", "use", "in", "on", "for", "and", "or", "with",
}


caps = [json.loads(line) for line in Path("outputs/rag_assets/english_pic_captions.jsonl").open(encoding="utf-8")]
for rec in caps:
    rec["_text"] = " ".join(
        str(rec.get(key, "") or "")
        for key in ["manual_id", "product", "caption_en", "nearest_section", "section_path", "image_id"]
    ).lower()
    rec["_tokens"] = collections.Counter(tokenize(rec["_text"]))


def score(question: str, rec: dict) -> float:
    q_tokens = collections.Counter(tokenize(question.lower()))
    value = 0.0
    for token, count in q_tokens.items():
        if token in STOP:
            continue
        if rec["_tokens"].get(token):
            value += (2.0 + min(count, 3)) * min(rec["_tokens"][token], 3)
    q_lower = question.lower()
    text = rec["_text"]
    for phrase in re.findall(r"[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){1,4}", q_lower):
        if len(phrase) >= 6 and phrase in text:
            value += len(phrase.split()) * 4
    return value


questions = {
    row["id"]: row["question"].strip('"')
    for row in csv.DictReader(open(r"C:\Users\admin\Downloads\question_public (2).csv", encoding="utf-8-sig", newline=""))
}

for item_id in ["249", "259", "289", "313", "372", "411", "241", "243", "248", "400"]:
    q = questions[item_id]
    top = sorted(((score(q, rec), rec) for rec in caps), key=lambda item: item[0], reverse=True)[:8]
    print("\n====", item_id, q)
    for value, rec in top:
        print(round(value, 1), rec["image_id"], rec["manual_id"], rec["product"], "::", rec["caption_en"][:180])
