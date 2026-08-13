from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--release", default="phase1-modular-core")
    parser.add_argument("--output", required=True)
    parser.add_argument("--docker-digest", default=os.environ.get("DOCKER_IMAGE_DIGEST", ""))
    parser.add_argument("--regression-summary", default="")
    parser.add_argument("--performance-report", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tracked = {
        "agent_api": root / "work" / "agent_api.py",
        "fastapi_server": root / "work" / "fastapi_server.py",
        "hybrid_manifest": root / "outputs" / "rag_assets" / "hybrid_index_v3" / "manifest.json",
        "hybrid_embeddings": root / "outputs" / "rag_assets" / "hybrid_index_v3" / "embeddings.npy",
        "hybrid_records": root / "outputs" / "rag_assets" / "hybrid_index_v3" / "records.jsonl",
        "manual_chunks": root / "outputs" / "rag_assets" / "manual_chunks.jsonl",
        "pic_mapping": root / "outputs" / "rag_assets" / "pic_mapping.jsonl",
        "image_knowledge": root / "outputs" / "rag_assets" / "image_knowledge_auto.jsonl",
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, raw_path in {
        "regression": args.regression_summary,
        "performance": args.performance_report,
    }.items():
        path = Path(raw_path).resolve() if raw_path else None
        if path and path.is_file():
            reports[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    commit = git_value(root, "rev-parse", "HEAD")
    exact_tag = git_value(root, "describe", "--tags", "--exact-match")
    manifest = {
        "release": args.release,
        "created_at": int(time.time()),
        "python": sys.version,
        "platform": platform.platform(),
        "source": {
            "git_commit": commit or None,
            "git_tag": exact_tag or None,
            "dirty": bool(git_value(root, "status", "--porcelain")),
        },
        "container_image": {
            "digest": args.docker_digest or None,
            "status": "built" if args.docker_digest else "pending_ci_build",
            "reason": "" if args.docker_digest else "Docker/BuildKit is unavailable on this GPU host.",
        },
        "versions": {
            "application": os.environ.get("APPLICATION_VERSION", "3.1.0-phase1"),
            "knowledge": os.environ.get("KNOWLEDGE_VERSION", "competition-kb-v1"),
            "model_configuration": os.environ.get("MODEL_CONFIGURATION_VERSION", "model-config-v1"),
            "prompt": os.environ.get("PROMPT_VERSION", "prompt-v1"),
        },
        "assets": {
            name: {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in tracked.items()
            if path.is_file()
        },
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
