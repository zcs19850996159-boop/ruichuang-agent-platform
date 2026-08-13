from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_retriever import mean_pool, tokenize_for_bm25


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs" / "rag_assets"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def language_of(text: str) -> str:
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    if cjk and letters:
        return "mixed"
    return "zh" if cjk else "en"


def hash_record(record: dict[str, Any]) -> str:
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_records(auto_image_knowledge: Path | None = None) -> list[dict[str, Any]]:
    pic_rows = [
        row for row in load_jsonl(ASSET_DIR / "pic_mapping.jsonl")
        if row.get("mapping_source") == "human_review" and row.get("image_id")
    ]
    reviewed_images = {str(row["image_id"]) for row in pic_rows}
    records: list[dict[str, Any]] = []
    chunks = load_jsonl(ASSET_DIR / "manual_chunks.jsonl") + load_jsonl(ASSET_DIR / "manual_chunks_extra.jsonl")
    seen: set[str] = set()
    for row in chunks:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        text = compact(
            " ".join(
                str(row.get(key) or "")
                for key in ("product", "section_title", "text", "captions")
            )
        )
        if not text:
            continue
        image_ids = [str(x) for x in (row.get("image_ids") or []) if str(x)]
        core = {
            "record_type": "chunk",
            "chunk_id": chunk_id,
            "manual_id": str(row.get("manual_id") or ""),
            "product": str(row.get("product") or ""),
            "section_title": compact(row.get("section_title") or ""),
            "section_path": compact(row.get("section_path") or row.get("section_title") or ""),
            "source_file": str(row.get("source_text_file") or ""),
            "source_page": row.get("source_page"),
            "language": language_of(text),
            "review_status": "human_reviewed_image_context" if any(x in reviewed_images for x in image_ids) else "source_extracted",
            "image_ids": image_ids,
            "text": text[:4000],
        }
        source_hash = hash_record(core)
        core["source_hash"] = source_hash
        core["chunk_version"] = "hybrid-v1-" + source_hash[:12]
        records.append(core)
    for row in pic_rows:
        image_id = str(row.get("image_id") or "")
        text = compact(
            " ".join(
                str(row.get(key) or "")
                for key in ("product", "caption", "section", "before_context", "after_context", "notes")
            )
        )
        if not text:
            continue
        core = {
            "record_type": "image",
            "chunk_id": f"image:{image_id}",
            "manual_id": str(row.get("manual_id") or ""),
            "product": str(row.get("product") or ""),
            "section_title": compact(row.get("section") or row.get("caption") or ""),
            "section_path": compact(row.get("section") or ""),
            "source_file": str(row.get("source_text_file") or ""),
            "source_page": row.get("source_page"),
            "language": language_of(text),
            "review_status": "human_reviewed",
            "image_ids": [image_id],
            "text": text[:2400],
        }
        source_hash = hash_record(core)
        core["source_hash"] = source_hash
        core["chunk_version"] = "hybrid-v1-" + source_hash[:12]
        records.append(core)
    auto_path = auto_image_knowledge or ASSET_DIR / "image_knowledge_auto.jsonl"
    for row in load_jsonl(auto_path):
        if str(row.get("status") or "ok") != "ok":
            continue
        image_id = str(row.get("image_id") or "")
        if not image_id:
            continue
        structured = row.get("structured") if isinstance(row.get("structured"), dict) else row
        text = compact(
            " ".join(
                [
                    str(row.get("product") or ""),
                    str(row.get("nearby_text") or ""),
                    str(structured.get("ocr_text") or ""),
                    str(structured.get("caption") or ""),
                    " ".join(str(x) for x in (structured.get("visual_tags") or [])),
                    " ".join(str(x) for x in (structured.get("components") or [])),
                    " ".join(str(x) for x in (structured.get("buttons") or [])),
                    " ".join(str(x) for x in (structured.get("error_codes") or [])),
                    str(structured.get("screen_state") or ""),
                ]
            )
        )
        if not text:
            continue
        core = {
            "record_type": "image_auto",
            "chunk_id": f"image-auto:{image_id}",
            "manual_id": str(row.get("manual_id") or ""),
            "product": str(row.get("product") or ""),
            "section_title": compact(row.get("section_title") or structured.get("caption") or ""),
            "section_path": compact(row.get("section_path") or row.get("section_title") or ""),
            "source_file": str(row.get("source") or row.get("path") or row.get("url") or ""),
            "source_page": row.get("source_page"),
            "language": language_of(text),
            "review_status": str(row.get("review_status") or "machine_generated"),
            "image_ids": [image_id],
            "text": text[:3000],
            "vision_confidence": structured.get("vision_confidence"),
            "ocr_confidence": structured.get("ocr_confidence"),
        }
        source_hash = hash_record(core)
        core["source_hash"] = source_hash
        core["chunk_version"] = "hybrid-v2-" + source_hash[:12]
        records.append(core)
    return records


def encode(records: list[dict[str, Any]], model_path: str, device: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if device.startswith("cuda"):
        model.half()
    vectors: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        texts = ["passage: " + record["text"] for record in records[start : start + batch_size]]
        batch = tokenizer(texts, max_length=384, padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            pooled = mean_pool(output.last_hidden_state, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.float().cpu().numpy())
        print(json.dumps({"encoded": min(start + batch_size, len(records)), "total": len(records)}), flush=True)
    return np.concatenate(vectors, axis=0).astype(np.float16)


def encode_incremental(
    records: list[dict[str, Any]],
    model_path: str,
    device: str,
    batch_size: int,
    previous: Path | None,
) -> tuple[np.ndarray, dict[str, int]]:
    reused: dict[str, np.ndarray] = {}
    previous_count = 0
    if previous and (previous / "records.jsonl").exists() and (previous / "embeddings.npy").exists():
        old_records = load_jsonl(previous / "records.jsonl")
        old_embeddings = np.load(previous / "embeddings.npy", mmap_mode="r")
        previous_count = min(len(old_records), len(old_embeddings))
        for idx, record in enumerate(old_records[:previous_count]):
            source_hash = str(record.get("source_hash") or "")
            if source_hash:
                reused[source_hash] = np.asarray(old_embeddings[idx], dtype=np.float16)
    missing_records = [record for record in records if str(record.get("source_hash") or "") not in reused]
    new_vectors = (
        encode(missing_records, model_path, device, batch_size)
        if missing_records
        else np.empty((0, 0), dtype=np.float16)
    )
    if reused:
        embedding_dim = int(next(iter(reused.values())).shape[0])
    elif len(new_vectors):
        embedding_dim = int(new_vectors.shape[1])
    else:
        raise RuntimeError("No records or reusable embeddings were found")
    output = np.empty((len(records), embedding_dim), dtype=np.float16)
    encoded_idx = 0
    reused_count = 0
    for idx, record in enumerate(records):
        old = reused.get(str(record.get("source_hash") or ""))
        if old is not None and int(old.shape[0]) == embedding_dim:
            output[idx] = old
            reused_count += 1
        else:
            output[idx] = new_vectors[encoded_idx]
            encoded_idx += 1
    return output, {
        "previous_count": previous_count,
        "reused_count": reused_count,
        "encoded_count": len(records) - reused_count,
        "removed_count": max(0, previous_count - reused_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/multilingual-e5-small")
    parser.add_argument("--output", default=str(ASSET_DIR / "hybrid_index_v1"))
    parser.add_argument("--previous", default="")
    parser.add_argument("--auto-image-knowledge", default=str(ASSET_DIR / "image_knowledge_auto.jsonl"))
    parser.add_argument("--index-version", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output)
    index_version = args.index_version.strip() or (
        "hybrid-v3" if output.name.endswith("_v3") else "hybrid-v2"
    )
    previous = Path(args.previous) if args.previous else (output if output.exists() else None)
    staging = output.parent / f".{output.name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=False)
    records = make_records(Path(args.auto_image_knowledge))
    embeddings, incremental = encode_incremental(
        records,
        args.model,
        args.device,
        max(1, args.batch_size),
        previous,
    )
    np.save(staging / "embeddings.npy", embeddings)
    (staging / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (staging / "bm25_tokens.txt").write_text(
        "\n".join(" ".join(tokenize_for_bm25(record["text"])) for record in records) + "\n",
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(
        "".join(record["source_hash"] for record in records).encode("utf-8")
    ).hexdigest()
    manifest = {
        "index_version": index_version,
        "embedding_model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "record_count": len(records),
        "chunk_count": sum(record["record_type"] == "chunk" for record in records),
        "image_count": sum(record["record_type"] in {"image", "image_auto"} for record in records),
        "auto_image_count": sum(record["record_type"] == "image_auto" for record in records),
        "source_hash": source_hash,
        "created_at": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 2),
        "incremental": incremental,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        old_output = output.parent / f".{output.name}.previous-{uuid.uuid4().hex[:8]}"
        output.replace(old_output)
        staging.replace(output)
        shutil.rmtree(old_output)
    else:
        staging.replace(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
