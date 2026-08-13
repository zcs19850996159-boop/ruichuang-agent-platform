from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def unwrap_image_features(value: Any) -> Any:
    """Support both tensor-returning and Transformers 5 model-output APIs."""
    if hasattr(value, "image_embeds") and value.image_embeds is not None:
        return value.image_embeds
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
        return value.last_hidden_state[:, 0]
    return value


def manual_id_from_image_id(image_id: str) -> str:
    prefix = str(image_id).split("_", 1)[0]
    return prefix if prefix.lower().startswith("manual") and prefix[6:].isdigit() else ""


class VisualImageMatcher:
    """Optional image-to-manual-image matcher backed by a normalized CLIP index."""

    def __init__(self) -> None:
        self.model_id = os.environ.get("VISUAL_RETRIEVAL_MODEL", DEFAULT_MODEL)
        self.index_path = ROOT / os.environ.get(
            "VISUAL_RETRIEVAL_INDEX",
            "outputs/rag_assets/visual_clip_index_v1.npz",
        )
        self.manifest_path = ROOT / os.environ.get(
            "VISUAL_RETRIEVAL_MANIFEST",
            "outputs/rag_assets/manual_image_manifest.jsonl",
        )
        self.records_path = ROOT / os.environ.get(
            "VISUAL_RETRIEVAL_RECORDS",
            "outputs/rag_assets/hybrid_index_v3/records.jsonl",
        )
        self.curated_evidence_path = ROOT / os.environ.get(
            "VISUAL_RETRIEVAL_CURATED_EVIDENCE",
            "assets/human_reviewed_image_evidence_v1.json",
        )
        self.enabled = os.environ.get(
            "VISUAL_RETRIEVAL_ENABLED",
            "1" if self.index_path.is_file() else "0",
        ) == "1"
        self.accept_score = float(os.environ.get("VISUAL_RETRIEVAL_ACCEPT_SCORE", "0.84"))
        self.accept_gap = float(os.environ.get("VISUAL_RETRIEVAL_ACCEPT_GAP", "0.015"))
        self.top_k = max(1, int(os.environ.get("VISUAL_RETRIEVAL_TOP_K", "5")))
        self._lock = threading.RLock()
        self._ready = False
        self._error = ""
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._embeddings: np.ndarray | None = None
        self._image_ids: list[str] = []
        self._metadata_by_image_id: dict[str, dict[str, str]] = {}
        self._manual_by_image_id = self._load_manual_map()
        self._curated_image_ids: set[str] = set()
        self._evidence_by_image_id = self._load_human_reviewed_evidence()

    def _load_manual_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if not self.manifest_path.is_file():
            return result
        try:
            with self.manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    image_id = str(row.get("image_id") or "").strip()
                    manual_id = str(row.get("manual_id") or "").strip()
                    if image_id and manual_id:
                        result.setdefault(image_id, manual_id)
                        self._metadata_by_image_id.setdefault(
                            image_id,
                            {
                                "product": str(row.get("product") or "").strip(),
                                "section_title": str(row.get("section_title") or "").strip(),
                                "section_path": str(row.get("section_path") or "").strip(),
                                "nearby_text": str(row.get("nearby_text") or "").strip(),
                            },
                        )
        except Exception:
            return {}
        return result

    def _manual_id(self, image_id: str) -> str:
        return self._manual_by_image_id.get(str(image_id), "") or manual_id_from_image_id(image_id)

    def _load_human_reviewed_evidence(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.records_path.is_file():
            try:
                with self.records_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("review_status") != "human_reviewed_image_context":
                            continue
                        text = " ".join(str(row.get("text") or "").split())
                        if not text:
                            continue
                        for image_id in row.get("image_ids") or []:
                            image_id = str(image_id)
                            if image_id and len(text) > len(result.get(image_id, "")):
                                result[image_id] = text[:1800]
            except Exception:
                result = {}
        if self.curated_evidence_path.is_file():
            try:
                payload = json.loads(self.curated_evidence_path.read_text(encoding="utf-8"))
                for image_id, item in (payload.get("images") or {}).items():
                    if isinstance(item, dict):
                        text = str(item.get("evidence") or "")
                    else:
                        text = str(item or "")
                    text = " ".join(text.split())
                    if image_id and text:
                        image_id = str(image_id)
                        result[image_id] = text[:1800]
                        self._curated_image_ids.add(image_id)
            except Exception:
                pass
        return result

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.index_path.is_file(),
            "ready": self._ready,
            "model": self.model_id,
            "index": str(self.index_path.relative_to(ROOT)) if self.index_path.is_relative_to(ROOT) else str(self.index_path),
            "image_count": len(self._image_ids),
            "mapped_image_count": len(self._manual_by_image_id),
            "evidence_image_count": len(self._evidence_by_image_id),
            "curated_evidence_image_count": len(self._curated_image_ids),
            "device": self._device,
            "error": self._error,
        }

    def _ensure_loaded(self) -> bool:
        if self._ready:
            return True
        if not self.enabled or not self.index_path.is_file():
            return False
        with self._lock:
            if self._ready:
                return True
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                payload = np.load(self.index_path, allow_pickle=False)
                self._embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
                self._image_ids = [str(x) for x in payload["image_ids"].tolist()]
                indexed_model = str(payload["model_id"].tolist())
                if indexed_model != self.model_id:
                    raise RuntimeError(f"index model {indexed_model!r} != configured model {self.model_id!r}")
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._processor = CLIPProcessor.from_pretrained(self.model_id, local_files_only=True)
                self._model = CLIPModel.from_pretrained(self.model_id, local_files_only=True).to(self._device).eval()
                self._ready = True
                self._error = ""
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"[:500]
                self._ready = False
            return self._ready

    def match(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "used": False,
            "accepted": False,
            "model": self.model_id,
            "matches": [],
            "top_score": 0.0,
            "top_gap": 0.0,
            "note": "",
        }
        if not images:
            return result
        if not self._ensure_loaded():
            result["note"] = self._error or "visual vector index is disabled or unavailable"
            return result
        try:
            import torch

            pil_images = []
            for item in images:
                raw = base64.b64decode(str(item.get("base64") or ""))
                pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            with self._lock, torch.inference_mode():
                inputs = self._processor(images=pil_images, return_tensors="pt")
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                features = unwrap_image_features(self._model.get_image_features(**inputs))
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                query = features.mean(dim=0)
                query = query / query.norm().clamp_min(1e-12)
                query_np = query.detach().float().cpu().numpy()
            scores = self._embeddings @ query_np
            order = np.argsort(-scores)[: max(self.top_k, 24)]
            if len(order):
                raw_top_index = int(order[0])
                raw_top_id = self._image_ids[raw_top_index]
                raw_top_manual = self._manual_id(raw_top_id)
                raw_top_score = float(scores[raw_top_index])
                curated_delta = float(
                    os.environ.get("VISUAL_CURATED_RERANK_MAX_DELTA", "0.025")
                )
                curated_candidates = [
                    int(index)
                    for index in order
                    if self._image_ids[int(index)] in self._curated_image_ids
                    and self._manual_id(self._image_ids[int(index)]) == raw_top_manual
                    and raw_top_score - float(scores[int(index)]) <= curated_delta
                ]
                if curated_candidates:
                    curated_top = max(
                        curated_candidates,
                        key=lambda index: float(scores[index]),
                    )
                    order = np.asarray(
                        [curated_top]
                        + [int(index) for index in order if int(index) != curated_top],
                        dtype=order.dtype,
                    )
            matches = [
                {
                    "image_id": self._image_ids[int(index)],
                    "manual_id": self._manual_id(self._image_ids[int(index)]),
                    "score": round(float(scores[int(index)]), 6),
                    "curated": self._image_ids[int(index)] in self._curated_image_ids,
                    "evidence_excerpt": self._evidence_by_image_id.get(
                        self._image_ids[int(index)], ""
                    ),
                    **self._metadata_by_image_id.get(self._image_ids[int(index)], {}),
                }
                for index in order[: self.top_k]
            ]
            top_score = float(matches[0]["score"]) if matches else 0.0
            top_manual = str(matches[0].get("manual_id") or "") if matches else ""
            competing_manual_score = 0.0
            if top_manual:
                for index in order[1:]:
                    candidate_id = self._image_ids[int(index)]
                    candidate_manual = self._manual_id(candidate_id)
                    if candidate_manual and candidate_manual != top_manual:
                        competing_manual_score = float(scores[int(index)])
                        break
            top_gap = top_score - competing_manual_score if top_manual else 0.0
            curated_high_confidence = (
                bool(matches)
                and str(matches[0].get("image_id") or "") in self._curated_image_ids
                and top_score
                >= float(os.environ.get("VISUAL_CURATED_ACCEPT_SCORE", "0.97"))
                and top_gap
                >= float(os.environ.get("VISUAL_CURATED_ACCEPT_GAP", "0.005"))
            )
            accepted = bool(top_manual) and (
                (top_score >= self.accept_score and top_gap >= self.accept_gap)
                or curated_high_confidence
            )
            result.update(
                {
                    "used": True,
                    "accepted": accepted,
                    "matches": matches,
                    "top_score": round(top_score, 6),
                    "top_gap": round(top_gap, 6),
                    "note": "visual match accepted for routing" if accepted else "visual matches are informational only",
                }
            )
        except Exception as exc:
            result["note"] = f"{type(exc).__name__}: {exc}"[:500]
        return result


def build_index(model_id: str, image_dir: Path, output_path: Path, batch_size: int) -> None:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not paths:
        raise SystemExit(f"no images found in {image_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    vectors: list[np.ndarray] = []
    valid_ids: list[str] = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images, ids = [], []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                ids.append(path.stem)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = unwrap_image_features(model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vectors.append(features.detach().float().cpu().numpy())
        valid_ids.extend(ids)
        print(json.dumps({"indexed": len(valid_ids), "total": len(paths)}, ensure_ascii=False), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.concatenate(vectors, axis=0).astype(np.float32),
        image_ids=np.asarray(valid_ids),
        model_id=np.asarray(model_id),
    )
    print(json.dumps({"output": str(output_path), "image_count": len(valid_ids), "model": model_id}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "assets" / "manual_images")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rag_assets" / "visual_clip_index_v1.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build_index(args.model, args.image_dir, args.output, max(1, args.batch_size))
