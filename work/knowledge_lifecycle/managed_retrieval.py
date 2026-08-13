from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
STRATEGY_VERSION = "ruichuang-managed-hybrid-v2"


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        if _CJK_RE.search(raw):
            if len(raw) <= 2:
                tokens.append(raw)
            else:
                tokens.extend(raw[index : index + 2] for index in range(len(raw) - 1))
                tokens.extend(raw[index : index + 3] for index in range(len(raw) - 2))
        elif len(raw) > 1:
            tokens.append(raw)
    return tokens


def _retrieval_text(record: dict[str, Any]) -> str:
    """Add local section context without changing the cited source text."""
    parts: list[str] = []
    seen: set[str] = set()
    for value in (
        record.get("title"),
        record.get("section"),
        record.get("text"),
    ):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        normalized = text.lower()
        if text and normalized not in seen:
            seen.add(normalized)
            parts.append(text)
    return "\n".join(parts)


def _source_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _model_configuration() -> tuple[str, str]:
    model = (
        os.environ.get("MANAGED_HYBRID_EMBEDDING_MODEL", "").strip()
        or os.environ.get("HYBRID_EMBEDDING_MODEL", "").strip()
    )
    if not model:
        server_default = Path(
            "/root/autodl-tmp/models/multilingual-e5-small"
        )
        if server_default.is_dir():
            model = str(server_default)
    device = (
        os.environ.get("MANAGED_HYBRID_DEVICE", "").strip()
        or os.environ.get("HYBRID_DEVICE", "cpu").strip()
        or "cpu"
    )
    return model, device


def _encode_passages(
    records: list[dict[str, Any]],
    *,
    model_path: str,
    device: str,
) -> Any:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if device.startswith("cuda"):
        model.half()
    vectors: list[Any] = []
    batch_size = max(1, int(os.environ.get("MANAGED_HYBRID_BATCH_SIZE", "32")))
    for start in range(0, len(records), batch_size):
        texts = [
            "passage: " + _retrieval_text(item)
            for item in records[start : start + batch_size]
        ]
        batch = tokenizer(
            texts,
            max_length=384,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            mask = batch["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float16)


def build_managed_index(
    snapshot_path: str | Path,
    *,
    index_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(snapshot_path)
    chunks_path = root / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if chunks_path.is_file() else []
    records = [
        {
            "record_type": "text_chunk",
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "document_id": str(chunk.get("document_id") or ""),
            "product_id": str(chunk.get("product_id") or ""),
            "title": str(chunk.get("title") or ""),
            "section": str(chunk.get("section") or ""),
            "page": chunk.get("page"),
            "chunk_type": str(chunk.get("chunk_type") or ""),
            "source_ref": str(chunk.get("source_ref") or ""),
            "image_ids": list(chunk.get("image_ids") or []),
            "role": str(chunk.get("role") or ""),
            "language": str(chunk.get("language") or ""),
            "model": str(chunk.get("model") or ""),
            "version": str(chunk.get("version") or ""),
            "text": str(chunk.get("text") or ""),
        }
        for chunk in chunks
        if str(chunk.get("text") or "").strip()
    ]
    manifest_path = root / "manifest.json"
    snapshot_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    for image in snapshot_manifest.get("images") or []:
        structured = image.get("structured") or {}
        image_text = " ".join(
            [
                str(structured.get("ocr_text") or ""),
                str(structured.get("caption") or ""),
                " ".join(str(item) for item in structured.get("visual_tags") or []),
                " ".join(str(item) for item in structured.get("components") or []),
                " ".join(str(item) for item in structured.get("buttons") or []),
                " ".join(str(item) for item in structured.get("error_codes") or []),
                str(structured.get("screen_state") or ""),
            ]
        ).strip()
        if not image_text:
            continue
        records.append(
            {
                "record_type": "manual_image",
                "chunk_id": f"image:{image.get('image_id', '')}",
                "document_id": str(image.get("document_id") or ""),
                "product_id": str(image.get("product_id") or ""),
                "title": str(image.get("source_name") or image.get("image_id") or ""),
                "section": str(structured.get("caption") or "")[:160],
                "page": image.get("source_page"),
                "chunk_type": "manual_image",
                "source_ref": f"image:{image.get('image_id', '')}",
                "image_ids": [str(image.get("image_id") or "")],
                "role": "manual_image",
                "language": "",
                "model": "",
                "version": str(image.get("chunk_version") or ""),
                "text": image_text,
            }
        )
    target_index = Path(index_dir) if index_dir is not None else root / "hybrid_index"
    target_index.mkdir(parents=True, exist_ok=True)
    (target_index / "records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    (target_index / "bm25_tokens.txt").write_text(
        "\n".join(
            " ".join(tokenize_for_bm25(_retrieval_text(item)))
            for item in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )

    model_path, device = _model_configuration()
    mode = "lexical_fallback"
    embedding_error = ""
    embedding_dim = 0
    embeddings_path = target_index / "embeddings.npy"
    embeddings_path.unlink(missing_ok=True)
    if records and model_path and Path(model_path).expanduser().is_dir():
        try:
            import numpy as np

            embeddings = _encode_passages(
                records,
                model_path=model_path,
                device=device,
            )
            np.save(embeddings_path, embeddings)
            embedding_dim = int(embeddings.shape[1])
            mode = "hybrid"
        except Exception as exc:
            embedding_error = f"{type(exc).__name__}: {exc}"[:500]

    manifest = {
        "schema_version": "1.0",
        "strategy_version": STRATEGY_VERSION,
        "mode": mode,
        "record_count": len(records),
        "source_hash": _source_hash(records),
        "embedding_model": model_path or None,
        "embedding_dim": embedding_dim,
        "device": device if mode == "hybrid" else None,
        "embedding_error": embedding_error or None,
        "weights": (
            {"vector": 0.55, "bm25": 0.35, "exact": 0.10}
            if mode == "hybrid"
            else {"bm25": 0.82, "exact": 0.18}
        ),
    }
    (target_index / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


class ManagedHybridRetriever:
    def __init__(
        self,
        snapshot_path: str | Path,
        *,
        index_dir: str | Path | None = None,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.index_dir = (
            Path(index_dir)
            if index_dir is not None
            else self.snapshot_path / "hybrid_index"
        )
        if not (self.index_dir / "manifest.json").is_file():
            build_managed_index(
                self.snapshot_path,
                index_dir=self.index_dir,
            )
        self.manifest = json.loads(
            (self.index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.records = [
            json.loads(line)
            for line in (self.index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        token_lines = (
            (self.index_dir / "bm25_tokens.txt").read_text(encoding="utf-8").splitlines()
            if (self.index_dir / "bm25_tokens.txt").is_file()
            else []
        )
        self.doc_tokens = [
            token_lines[index].split() if index < len(token_lines) else []
            for index in range(len(self.records))
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        )
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.document_frequency.update(set(tokens))
        self.embeddings = None
        if self.manifest.get("mode") == "hybrid":
            try:
                import numpy as np

                self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
            except Exception:
                self.manifest = {**self.manifest, "mode": "lexical_fallback"}
        self._model = None
        self._tokenizer = None
        self._model_lock = threading.RLock()

    def _bm25(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.records)
        if not query_tokens:
            return scores
        query_counts = Counter(query_tokens)
        document_count = max(1, len(self.records))
        for index, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            length = max(1.0, float(len(tokens)))
            for token, query_count in query_counts.items():
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                doc_frequency = self.document_frequency.get(token, 0)
                inverse_frequency = math.log(
                    1.0 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * length / max(self.average_length, 1.0)
                )
                scores[index] += (
                    inverse_frequency
                    * (frequency * 2.5 / denominator)
                    * min(query_count, 2)
                )
        return scores

    def _encode_query(self, query: str) -> Any:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = str(self.manifest.get("embedding_model") or "")
        device = str(self.manifest.get("device") or "cpu")
        with self._model_lock:
            if self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
                self._model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                ).eval().to(device)
                if device.startswith("cuda"):
                    self._model.half()
            batch = self._tokenizer(
                ["query: " + re.sub(r"\s+", " ", query).strip()],
                max_length=384,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self._model(**batch)
                mask = batch["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled[0].float().cpu().numpy()

    def _expand_section_companions(
        self,
        ranked: list[tuple[float, int, float]],
        question: str,
        *,
        top_k: int,
    ) -> list[tuple[float, int, float]]:
        """Keep the richest body evidence beside an explicitly requested section."""
        limit = max(1, min(int(top_k), 20))
        selected = list(ranked[:limit])
        selected_indexes = {item[1] for item in selected}
        normalized_question = re.sub(r"\s+", "", question).lower()
        anchor_keys: list[tuple[str, str]] = []
        for _, index, _ in selected:
            record = self.records[index]
            section = re.sub(r"\s+", "", str(record.get("section") or "")).lower()
            section_without_number = re.sub(
                r"^\d+(?:\.\d+){0,4}[、.)）:：-]*",
                "",
                section,
            )
            aliases = [
                alias
                for alias in (section, section_without_number)
                if len(alias) >= 2
            ]
            key = (
                str(record.get("document_id") or ""),
                str(record.get("section") or ""),
            )
            if (
                any(alias in normalized_question for alias in aliases)
                and key not in anchor_keys
            ):
                anchor_keys.append(key)

        for document_id, section in anchor_keys:
            candidates = [
                item
                for item in ranked
                if str(self.records[item[1]].get("document_id") or "") == document_id
                and str(self.records[item[1]].get("section") or "") == section
                and str(self.records[item[1]].get("chunk_type") or "")
                not in {"heading", "manual_image"}
            ]
            companion = max(
                candidates,
                key=lambda item: (
                    str(self.records[item[1]].get("chunk_type") or "")
                    in {"paragraph", "table", "error_code"},
                    min(len(str(self.records[item[1]].get("text") or "")), 1200),
                    item[0],
                ),
                default=None,
            )
            if companion is None:
                continue
            anchor_position = next(
                (
                    position
                    for position, item in enumerate(selected)
                    if str(self.records[item[1]].get("document_id") or "")
                    == document_id
                    and str(self.records[item[1]].get("section") or "")
                    == section
                    and item[1] != companion[1]
                ),
                None,
            )
            if companion[1] in selected_indexes:
                if anchor_position is not None:
                    companion_position = next(
                        position
                        for position, item in enumerate(selected)
                        if item[1] == companion[1]
                    )
                    item = selected.pop(companion_position)
                    if companion_position < anchor_position:
                        anchor_position -= 1
                    selected.insert(anchor_position + 1, item)
                continue
            replacement = next(
                (
                    position
                    for position in range(len(selected) - 1, -1, -1)
                    if not (
                        str(self.records[selected[position][1]].get("document_id") or "")
                        == document_id
                        and str(self.records[selected[position][1]].get("section") or "")
                        == section
                    )
                ),
                None,
            )
            if replacement is None:
                replacement = len(selected) - 1
            selected_indexes.discard(selected[replacement][1])
            selected[replacement] = companion
            selected_indexes.add(companion[1])
            if anchor_position is not None:
                companion_position = next(
                    position
                    for position, item in enumerate(selected)
                    if item[1] == companion[1]
                )
                item = selected.pop(companion_position)
                if companion_position < anchor_position:
                    anchor_position -= 1
                selected.insert(anchor_position + 1, item)
        return selected

    def search(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        if not self.records:
            return {
                "hits": [],
                "evidence_sufficient": False,
                "retrieval": self.manifest,
            }
        query_tokens = tokenize_for_bm25(question)
        bm25 = self._bm25(query_tokens)
        normalized_question = re.sub(r"\s+", "", question).lower()
        exact = []
        for record in self.records:
            text = re.sub(r"\s+", "", _retrieval_text(record)).lower()
            product = str(record.get("product_id") or "").lower()
            model = str(record.get("model") or "").lower()
            phrase = 1.0 if normalized_question and normalized_question in text else 0.0
            metadata_match = 1.0 if (product and product in question.lower()) or (model and model in question.lower()) else 0.0
            exact.append(max(phrase, metadata_match))

        vector_scores = None
        mode = str(self.manifest.get("mode") or "lexical_fallback")
        if mode == "hybrid" and self.embeddings is not None:
            try:
                import numpy as np

                vector_scores = np.asarray(self.embeddings, dtype=np.float32) @ self._encode_query(question)
            except Exception:
                vector_scores = None
                mode = "lexical_fallback"

        ranked: list[tuple[float, int, float]] = []
        for index, bm25_score in enumerate(bm25):
            bm25_norm = bm25_score / (bm25_score + 12.0) if bm25_score > 0 else 0.0
            if vector_scores is not None:
                vector = float(vector_scores[index])
                vector_norm = min(1.0, max(0.0, (vector - 0.55) / 0.40))
                score = 0.55 * vector_norm + 0.35 * bm25_norm + 0.10 * exact[index]
            else:
                vector = 0.0
                score = 0.82 * bm25_norm + 0.18 * exact[index]
            if score > 0:
                ranked.append((score, index, vector))
        ranked.sort(key=lambda item: (-item[0], self.records[item[1]]["chunk_id"]))
        selected = self._expand_section_companions(
            ranked,
            question,
            top_k=top_k,
        )
        hits = []
        for rank, (score, index, vector) in enumerate(selected, start=1):
            hits.append(
                {
                    **self.records[index],
                    "rank": rank,
                    "score": round(score, 6),
                    "bm25_score": round(bm25[index], 6),
                    "vector_score": round(vector, 6) if vector_scores is not None else None,
                    "exact_score": exact[index],
                }
            )
        top_score = float(hits[0]["score"]) if hits else 0.0
        threshold = 0.08 if mode == "hybrid" else 0.04
        return {
            "hits": hits,
            "evidence_sufficient": bool(hits) and top_score >= threshold,
            "retrieval": {
                **self.manifest,
                "mode": mode,
                "threshold": threshold,
            },
        }


_CACHE: dict[tuple[str, int], ManagedHybridRetriever] = {}
_CACHE_LOCK = threading.RLock()


def search_snapshot(
    snapshot_path: str | Path,
    question: str,
    *,
    top_k: int = 5,
    index_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(snapshot_path).resolve()
    target_index = (
        Path(index_dir).resolve()
        if index_dir is not None
        else root / "hybrid_index"
    )
    manifest_path = target_index / "manifest.json"
    if not manifest_path.is_file():
        build_managed_index(root, index_dir=target_index)
    cache_key = (
        f"{root}::{target_index}",
        manifest_path.stat().st_mtime_ns,
    )
    with _CACHE_LOCK:
        retriever = _CACHE.get(cache_key)
        if retriever is None:
            retriever = ManagedHybridRetriever(
                root,
                index_dir=target_index,
            )
            _CACHE.clear()
            _CACHE[cache_key] = retriever
    return retriever.search(question, top_k=top_k)
