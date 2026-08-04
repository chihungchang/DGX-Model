# Workspace notes

Last inspected: 2026-08-04 (Asia/Taipei)

This file is a concise operating map of `/home/chihung/ai`. Runtime status is
only a dated snapshot; profile files and Compose configuration remain the source
of truth.

## Top-level layout

| Path | Purpose | Persistent/runtime data |
| --- | --- | --- |
| `llama.cpp/` | GGUF LLM runtime and profile manager | Host models under `/home/chihung/models/gguf` |
| `vllm/` | vLLM OpenAI-compatible server and profile manager | Host models under `/home/chihung/models/huggingface` |

The workspace root is tracked in the `chihungchang/DGX-Model` Git repository
and contains no `AGENTS.md`.

## Runtime architecture and ports

- vLLM and llama.cpp are alternative main-LLM runtimes. The root `dgx-model`
  command is their only manager and safely switches between them because DGX
  Spark uses unified CPU/GPU memory.
- Both vLLM and llama.cpp expose their mutually exclusive OpenAI-compatible API
  on the fixed port `8080`.
- llama.cpp also exposes its web UI, health endpoint, and metrics on port `8080`.

## Dated runtime snapshot

At inspection time:

- `dgx-llamacpp` was running and healthy with profile
  `qwen36-fable-q4-mtp` (Qwen 3.6 27B Fable Fusion Q4_K_M, MTP,
  multimodal, 262,144 context).
- `dgx-vllm` did not exist and was not running.

## Configuration sources

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
- Only `llama.cpp/Dockerfile`, `llama.cpp/compose.yaml`, and
  `llama.cpp/profiles/*.args` are configuration sources. Backup paths are
  ignored and must not be used as sources.

### vLLM

- Lifecycle manager: root `dgx-model` command
- Shared deployment: `vllm/compose.yaml`
- Safe argument-array entrypoint: `vllm/launch.sh`
- Image is pinned by digest.
- Current profiles:
  - `qwen`: Qwen3.6 27B NVFP4, MTP=2, 262,144 context, GPU utilization 0.65
  - `qwen-mtp2-cosvc`: co-residency baseline, effectively similar to `qwen`
  - `qwen-dflash`: Qwen3.6 27B NVFP4 + DFlash, GPU utilization 0.4
  - `qwen35`: Qwen3.6 35B A3B NVFP4, FP8 KV, FlashInfer, MTP=3
  - `qwen35-q4kv`: same 35B checkpoint, INT4 KV, Triton attention, MTP=2
- Only `vllm/compose.yaml` and direct `vllm/profiles/*.args` files are
  deployable configuration sources. Historical tuning candidates are not kept
  in the workspace, and backups are ignored.

## Documentation note

The root `README.md` is the authoritative command reference. Backend READMEs
describe their current installed profiles.

## Safe operating assumptions

- Preserve both external model roots unless deletion is explicitly requested.
- Treat `profiles/*.args` as authoritative for model/runtime tuning.
- Use `dgx-model use`; it validates the target and gracefully stops the other
  main engine before switching.
- Recheck manager status before operational changes; the snapshot above will
  become stale.
