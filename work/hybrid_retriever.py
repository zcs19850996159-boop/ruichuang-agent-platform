from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        if _CJK_RE.search(raw):
            if len(raw) <= 2:
                tokens.append(raw)
            else:
                tokens.extend(raw[i : i + 2] for i in range(len(raw) - 1))
                tokens.extend(raw[i : i + 3] for i in range(len(raw) - 2))
        elif len(raw) > 1:
            tokens.append(raw)
    return tokens


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class HybridRetriever:
    def __init__(self, index_dir: Path, model_path: str, device: str = "cpu") -> None:
        self.index_dir = Path(index_dir)
        self.model_path = str(model_path)
        self.device = device
        self.manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.records = [
            json.loads(line)
            for line in (self.index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self.doc_tokens = [
            line.split(" ") if line else []
            for line in (self.index_dir / "bm25_tokens.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.doc_lens = np.asarray([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avg_doc_len = float(self.doc_lens.mean()) if len(self.doc_lens) else 1.0
        self.df = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))
        self.manual_indices: dict[str, np.ndarray] = {}
        grouped: dict[str, list[int]] = {}
        for idx, record in enumerate(self.records):
            grouped.setdefault(str(record.get("manual_id") or ""), []).append(idx)
        for manual_id, indices in grouped.items():
            self.manual_indices[manual_id] = np.asarray(indices, dtype=np.int32)
        self._model = None
        self._tokenizer = None
        self._model_lock = threading.RLock()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "index_version": self.manifest.get("index_version"),
            "records": len(self.records),
            "embedding_model": self.manifest.get("embedding_model"),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "device": self.device,
        }

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            self._model.eval()
            self._model.to(self.device)
            if self.device.startswith("cuda"):
                self._model.half()

    def encode_query(self, query: str) -> np.ndarray:
        import torch

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        text = "query: " + re.sub(r"\s+", " ", str(query or "")).strip()
        with self._model_lock, torch.inference_mode():
            batch = self._tokenizer(
                [text],
                max_length=384,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self._model(**batch)
            pooled = mean_pool(output.last_hidden_state, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled[0].float().cpu().numpy()

    def _bm25_scores(self, query_tokens: list[str], indices: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(indices), dtype=np.float32)
        if not query_tokens or not len(indices):
            return scores
        n_docs = max(1, len(self.records))
        q_counts = Counter(query_tokens)
        k1 = 1.5
        b = 0.75
        for pos, doc_idx in enumerate(indices):
            tokens = self.doc_tokens[int(doc_idx)]
            counts = Counter(tokens)
            dl = max(1.0, float(len(tokens)))
            score = 0.0
            for token, q_count in q_counts.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * dl / max(self.avg_doc_len, 1.0))
                score += idf * (tf * (k1 + 1.0) / denom) * min(q_count, 2)
            scores[pos] = score
        return scores

    def search(
        self,
        query: str,
        manual_id: str = "",
        top_k: int = 10,
        legacy_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        indices = self.manual_indices.get(str(manual_id or ""))
        if indices is None:
            indices = np.arange(len(self.records), dtype=np.int32)
        if not len(indices):
            return []
        query_vec = self.encode_query(query)
        vector_scores = np.asarray(self.embeddings[indices], dtype=np.float32) @ query_vec
        bm25_scores = self._bm25_scores(tokenize_for_bm25(query), indices)
        legacy_scores = legacy_scores or {}
        legacy_values = np.asarray(
            [float(legacy_scores.get(str(self.records[int(i)].get("chunk_id") or ""), 0.0)) for i in indices],
            dtype=np.float32,
        )
        # Absolute saturation keeps scores comparable across queries. Per-query max normalization would make one
        # accidental token overlap look as strong as an exact section-title match.
        vector_norm = np.clip((vector_scores - 0.55) / 0.40, 0.0, 1.0)
        bm25_norm = np.clip(bm25_scores / (bm25_scores + 12.0), 0.0, 1.0)
        legacy_norm = np.clip(legacy_values / (legacy_values + 40.0), 0.0, 1.0)
        rerank_scores = 0.48 * vector_norm + 0.37 * bm25_norm + 0.15 * legacy_norm
        order = np.argsort(-rerank_scores)[: max(1, int(top_k))]
        results: list[dict[str, Any]] = []
        for rank, pos in enumerate(order, 1):
            doc_idx = int(indices[int(pos)])
            record = dict(self.records[doc_idx])
            record.update(
                {
                    "rank": rank,
                    "vector_score": round(float(vector_scores[int(pos)]), 6),
                    "bm25_score": round(float(bm25_scores[int(pos)]), 6),
                    "legacy_score": round(float(legacy_values[int(pos)]), 6),
                    "rerank_score": round(float(rerank_scores[int(pos)]), 6),
                }
            )
            results.append(record)
        if results:
            second = float(results[1]["rerank_score"]) if len(results) > 1 else 0.0
            gap = round(float(results[0]["rerank_score"]) - second, 6)
            for record in results:
                record["top_score_gap"] = gap
        return results


_INSTANCE: HybridRetriever | None = None
_INSTANCE_LOCK = threading.RLock()


def get_hybrid_retriever(index_dir: Path, model_path: str, device: str = "cpu") -> HybridRetriever:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = HybridRetriever(index_dir=index_dir, model_path=model_path, device=device)
        return _INSTANCE


def hybrid_available(index_dir: Path) -> bool:
    required = ("manifest.json", "records.jsonl", "embeddings.npy", "bm25_tokens.txt")
    return all((Path(index_dir) / name).exists() for name in required)
