#!/usr/bin/env python3
"""Download the public model dependencies used by the production snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


QWEN_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
E5_REPO = "intfloat/multilingual-e5-small"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=QWEN_REPO,
        revision=QWEN_REVISION,
        local_dir=models_dir / "Qwen2.5-VL-3B-Instruct",
    )
    snapshot_download(
        repo_id=E5_REPO,
        local_dir=models_dir / "multilingual-e5-small",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    print(f"Models downloaded to {models_dir}")
    print("Use inventory.json from the production snapshot for exact SHA-256 comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
