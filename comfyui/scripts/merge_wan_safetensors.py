#!/usr/bin/env python3
"""Merge a Hugging Face sharded Wan diffusion model for ComfyUI's UNETLoader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    index_path = args.source / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    expected_keys = set(index["weight_map"])
    shard_names = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for position, shard_name in enumerate(shard_names, start=1):
        shard_path = args.source / shard_name
        print(
            f"[{position}/{len(shard_names)}] loading {shard_path.name}",
            flush=True,
        )
        shard = load_file(shard_path, device="cpu")
        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            raise RuntimeError(f"Duplicate tensor keys: {sorted(duplicate_keys)[:5]}")
        state_dict.update(shard)

    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(f"Index mismatch; missing={missing[:5]}, extra={extra[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    print(
        f"writing {len(state_dict)} tensors to {temporary_output}",
        flush=True,
    )
    save_file(state_dict, temporary_output, metadata={"format": "pt"})
    os.replace(temporary_output, args.output)
    print(f"complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
