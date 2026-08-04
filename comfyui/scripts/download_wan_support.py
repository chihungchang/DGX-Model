#!/usr/bin/env python3
"""Download the small ComfyUI-specific files needed by the Wan Animate workflow."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_ROOT = Path("/models")
DOWNLOAD_ROOT = MODEL_ROOT / ".downloads" / "wan-animate-support"

FILES = (
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "06e001fc51048fb03433a6fb25334de7836704a5",
        "split_files/clip_vision/clip_vision_h.safetensors",
        MODEL_ROOT / "clip_vision" / "clip_vision_h.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "06e001fc51048fb03433a6fb25334de7836704a5",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        MODEL_ROOT / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "8260d429d19fd7a72304cad059160b95d843913f",
        "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        MODEL_ROOT
        / "loras"
        / "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "fb1388adc906ab39ffc26ee40e96b22886b56bc4",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        MODEL_ROOT
        / "loras"
        / "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "8260d429d19fd7a72304cad059160b95d843913f",
        "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
        MODEL_ROOT
        / "loras"
        / "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors",
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
