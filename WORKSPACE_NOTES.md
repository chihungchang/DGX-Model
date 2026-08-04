# Workspace notes

Last inspected: 2026-08-04 (Asia/Taipei)

This file is a concise operating map of `/home/chihung/ai`. Runtime status is
only a dated snapshot; profile files and Compose configuration remain the source
of truth.

## Top-level layout

| Path | Purpose | Persistent/runtime data |
| --- | --- | --- |
| `audio/` | Qwen3 ASR + TTS deployment for Mazi Live Mode | Docker volume `mazi-audio-hf-cache` |
| `comfyui/` | ComfyUI image/video generation runtime | `comfyui/data/` (about 41 GiB allocated) |
| `llama.cpp/` | GGUF LLM runtime and profile manager | Host models under `/home/chihung/models/gguf` |
| `vllm/` | vLLM OpenAI-compatible server and profile manager | Host models under `/home/chihung/models/huggingface` |

The workspace root is tracked in the `chihungchang/DGX-Model` Git repository
and contains no `AGENTS.md`. The only nested repository found is
`comfyui/data/custom_nodes/comfyui_controlnet_aux`; it was clean and on a
detached HEAD when inspected.

## Runtime architecture and ports

- vLLM and llama.cpp are alternative main-LLM runtimes. The root `dgx-model`
  command is their only manager and safely switches between them because DGX
  Spark uses unified CPU/GPU memory.
- Both vLLM and llama.cpp expose their mutually exclusive OpenAI-compatible API
  on the fixed port `8080`.
- llama.cpp also exposes its web UI, health endpoint, and metrics on port `8080`.
- Audio ASR uses port `8101`; TTS uses port `8102`.
- ComfyUI uses port `8188` and stores all mutable data below `comfyui/data/`.

## Dated runtime snapshot

At inspection time:

- `dgx-llamacpp` was running and healthy with profile
  `qwen36-fable-q4-mtp` (Qwen 3.6 27B Fable Fusion Q4_K_M, MTP,
  multimodal, 262,144 context).
- `dgx-vllm` did not exist and was not running.
- `mazi-qwen3-asr` and `mazi-qwen3-tts` were both running and healthy.
- Audio profile `qwen-live-500ms` was active and validated; the manager reported
  the deployment as `ready`.
- ComfyUI was not running.

## Configuration sources

### Audio

- Lifecycle manager: `audio/audio-model`
- Shared deployment: `audio/compose.yaml`
- Runtime/profile parser: `audio/runtime/profile.py`
- ASR service shim: `audio/asr/server.py`
- Profiles are the only model/runtime configuration source:
  - `qwen-live.args`: 1.0-second ASR decode interval
  - `qwen-live-500ms.args`: 0.5-second ASR decode interval; currently active
- Both current profiles configure ASR GPU utilization `0.14`; TTS stages use
  `0.08` and `0.06`.
- Static and concurrent smoke tests live under `audio/scripts/`.

### llama.cpp

- Lifecycle manager: root `dgx-model` command
- Shared deployment: `llama.cpp/compose.yaml`
- Safe argument-array entrypoint: `llama.cpp/launch.sh`
- Image: locally built `mazi/llama.cpp:b10236-cuda13-openssl`, based on pinned
  llama.cpp commit `1464c62d88f699ec9700c8010bbfdbc603a9efd6`.
- Available profiles:
  - `qwen36-27b-q4-mtp-c4`: 1,048,576 context, parallel 4, external MTP draft
  - `qwen36-27b-q4-mtp`: 262,144 context, external MTP draft
  - `qwen36-27b-q4`: 262,144 context, no speculative draft
  - `qwen36-fable-q4-mtp`: 262,144 context, embedded MTP + mmproj; active
  - `qwythos`: 131,072 context, embedded MTP
- Historical backups are retained in `.backup-*` directories and
  `*.b10214-backup` files.

### vLLM

- Lifecycle manager: root `dgx-model` command
- Shared deployment: `vllm/compose.yaml`
- Safe argument-array entrypoint: `vllm/launch.sh`
- Image is pinned by digest.
- Current non-backup profiles:
  - `qwen`: Qwen3.6 27B NVFP4, MTP=2, 262,144 context, GPU utilization 0.65
  - `qwen-mtp2-cosvc`: co-residency baseline, effectively similar to `qwen`
  - `qwen-dflash`: Qwen3.6 27B NVFP4 + DFlash, GPU utilization 0.4
  - `qwen35`: Qwen3.6 35B A3B NVFP4, FP8 KV, FlashInfer, MTP=3
  - `qwen35-q4kv`: same 35B checkpoint, INT4 KV, Triton attention, MTP=2
- Benchmark/tuning notes are under `vllm/benchmarks/` and
  `vllm/profiles/tuning-20260720/`; prior configurations are under
  `vllm/backups/` and `*.bak` files.

### ComfyUI

- ComfyUI is pinned to commit `34d0629452cac83dc20aa3d84e45c9b60d9e36b3`
  (documented as v0.28.3) on NVIDIA PyTorch 25.11 ARM64.
- Persistent areas: `models/`, `custom_nodes/`, `input/`, `output/`, `temp/`,
  `user/`, and `cache/` below `comfyui/data/`.
- API workflows:
  - LTX 2.3 image-to-video (general and human variants)
  - Wan Animate
  - Wan S2V
  - Wan VACE
- Installed custom node: pinned `comfyui_controlnet_aux` repository.
- Most allocated model space is LTX 2.3 support: about 36 GiB under
  `models/.downloads`, 3.2 GiB of LoRAs, and a 950 MiB spatial upscaler.
  Destination model files are hard links to the download cache, so their
  apparent logical sizes do not consume a second copy on disk.
- Inputs occupy about 3 MiB and outputs about 1.7 MiB at inspection time.

## Documentation note

The root `README.md` is the authoritative command reference. Backend READMEs
describe their current installed profiles. The audio README still describes an
older ASR GPU limit of `0.08`; deployed audio profiles specify `0.14`.

## Safe operating assumptions

- Preserve everything under `comfyui/data/` and both external model roots
  unless deletion is explicitly requested.
- Treat `profiles/*.args` as authoritative for model/runtime tuning.
- Use `dgx-model use`; it validates the target and gracefully stops the other
  main engine before switching.
- Recheck manager status before operational changes; the snapshot above will
  become stale.
