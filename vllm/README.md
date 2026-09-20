# DGX Spark vLLM backend

This directory is the vLLM backend managed exclusively by
the repository-root `dgx-model` command. It serves an OpenAI-compatible API on
port `8080` and mounts checkpoints read-only from the host Hugging Face model
directory at `$HOME/models/huggingface`.

The container image is pinned by digest in `compose.yaml`. Profiles under
`profiles/*.args` are the only source of model-specific vLLM arguments.

## Installed profiles

| Profile | Checkpoint | Important settings |
| --- | --- | --- |
| `muse-glimmer` | `meta-models/Muse-Glimmer-30B` | BF16, 128K context, Muse reasoning/tool parsers, official pre-release image |
| `muse-glimmer-dflash` | `meta-models/Muse-Glimmer-30B` plus `meta-models/Muse-Glimmer-30B-assistant` | Official DFlash drafter, 15 speculative tokens, verified on DGX Spark |
| `qwen` | `nvidia/Qwen3.6-27B-NVFP4` | ModelOpt NVFP4, MTP=2, GPU utilization 0.65 |
| `qwen-mtp2-cosvc` | `nvidia/Qwen3.6-27B-NVFP4` | ASR/TTS co-residency baseline |
| `qwen-dflash` | `nvidia/Qwen3.6-27B-NVFP4` plus pinned `z-lab/Qwen3.6-27B-DFlash` | DFlash=12, GPU utilization 0.4 |
| `qwen35` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | FP8 KV, FlashInfer, MTP=3 |
| `qwen35-q4kv` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | INT4 KV, Triton attention, MTP=2 |

The Qwen profiles use a 262,144-token context. Muse Glimmer uses its native
131,072-token context. All profiles retain up to four concurrent sequences.

## Commands

```bash
./dgx-model list vllm
./dgx-model show vllm qwen-dflash
./dgx-model image pull vllm
./dgx-model image pull vllm muse-glimmer
./dgx-model model pull vllm qwen-dflash
./dgx-model validate vllm qwen-dflash
./dgx-model use vllm qwen-dflash
./dgx-model smoke vllm
./dgx-model logs vllm --follow
./dgx-model stop vllm
```

`model pull` uses the host `hf` CLI. No Hugging Face token is passed into the
serving container. The DFlash profile carries structured revision metadata so
its drafter is downloaded at the tested commit rather than an unpinned branch.

`launch.sh` revalidates the main model and every explicit/speculative model
directory before executing `vllm serve`.

Muse Glimmer support is not yet part of the general vLLM image. Its profile
pins the official `vllm/vllm-openai:muse-glimmer` multi-architecture image by
digest; other profiles continue to use the default pinned image in
`compose.yaml`. DFlash is intentionally disabled for this profile until its
DGX Spark startup path is fixed upstream. A BF16 cold start can approach or
exceed 15 minutes while loading the 55.5 GiB checkpoint across two shards, so
the shared startup health window is 30 minutes.

The separate `muse-glimmer-dflash` profile enables the official 5-layer DFlash
assistant and has been startup-, generation-, and acceptance-metric-verified on
DGX Spark. The non-speculative profile remains available as a fallback.

Only `compose.yaml` and direct `profiles/*.args` files are deployable
configuration sources. Historical tuning candidates are not kept in the
workspace; backup files and directories are ignored and must never be used as
runtime configuration sources.
