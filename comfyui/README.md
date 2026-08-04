# DGX Spark ComfyUI

ComfyUI engine for the DGX Spark. Model files, custom nodes, inputs, workflows,
and outputs remain outside the image under `data/`.

## Pinned runtime

- ComfyUI `v0.28.3`
- ComfyUI commit `34d0629452cac83dc20aa3d84e45c9b60d9e36b3`
- NVIDIA PyTorch `25.11-py3`, ARM64 digest
  `sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70`
- TorchAudio `2.9.1` ARM64 wheel installed without replacing NVIDIA PyTorch
- ARM64 dependencies required by the Wan Animate DWPose preprocessor
- Host port `8188`

## Wan Animate prototype

The prototype uses ComfyUI's native Wan Animate nodes and the pinned
`comfyui_controlnet_aux` custom node at commit
`e8b689a513c3e6b63edc44066560ca5919c0576e`.

The original Hugging Face Wan Animate shards are merged for `UNETLoader` with:

```bash
python scripts/merge_wan_safetensors.py \
  /path/to/Wan2.2-Animate-14B \
  /path/to/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors
```

ComfyUI-specific support weights are downloaded resumably with
`scripts/download_wan_support.py`.

## Persistent data

Everything under `data/` persists across container replacement:

- `models/`
- `custom_nodes/`
- `input/`
- `output/`
- `temp/`
- `user/`
- `cache/`

## Lifecycle

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f --tail 200
docker compose down
```

The LAN URL is `http://10.0.0.100:8188`.
