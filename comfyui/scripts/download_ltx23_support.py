#!/usr/bin/env python3
"""Download the official ComfyUI files needed by the LTX-2.3 I2V workflow."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_ROOT = Path("/models")
DOWNLOAD_ROOT = MODEL_ROOT / ".downloads" / "ltx-2.3-support"

FILES = (
    (
        "Lightricks/LTX-2.3-fp8",
        "1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1",
        "ltx-2.3-22b-dev-fp8.safetensors",
        MODEL_ROOT / "checkpoints" / "ltx-2.3-22b-dev-fp8.safetensors",
    ),
    (
        "Comfy-Org/ltx-2.3",
        "e14c0e7f46a1d68384214a3c6e6b309a382016dd",
        "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
        MODEL_ROOT
        / "loras"
        / "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    ),
    (
        "Comfy-Org/ltx-2",
        "bd5f9c87fcb0360ae7112f9784562670894d9492",
        "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
        MODEL_ROOT
        / "loras"
        / "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
    ),
    (
        "Comfy-Org/ltx-2",
        "bd5f9c87fcb0360ae7112f9784562670894d9492",
        "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        MODEL_ROOT / "text_encoders" / "gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    (
        "Lightricks/LTX-2.3",
        "4229404625088d21c4f112eb640fb04a0900ee25",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        MODEL_ROOT
        / "latent_upscale_models"
        / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ),
)


def link_download(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    temporary_destination.unlink(missing_ok=True)
    os.link(source, temporary_destination)
    os.replace(temporary_destination, destination)


def main() -> None:
    for repo_id, revision, filename, destination in FILES:
        if destination.exists():
            print(f"already present: {destination}", flush=True)
            continue
        print(f"downloading {repo_id}/{filename}", flush=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                local_dir=DOWNLOAD_ROOT / repo_id.replace("/", "--"),
            )
        )
        link_download(downloaded, destination)
        print(f"ready: {destination} ({destination.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
