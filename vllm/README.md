# DGX Spark vLLM backend

This directory is the vLLM backend managed exclusively by
the repository-root `dgx-model` command. It serves an OpenAI-compatible API on
port `8080` and mounts checkpoints read-only from the host Hugging Face model
directory configured by Compose.

The container image is pinned by digest in `compose.yaml`. Profiles under
`profiles/*.args` are the only source of model-specific vLLM arguments.

## Installed profiles

| Profile | Checkpoint | Important settings |
| --- | --- | --- |
| `qwen` | `nvidia/Qwen3.6-27B-NVFP4` | ModelOpt NVFP4, MTP=2, GPU utilization 0.65 |
| `qwen-mtp2-cosvc` | `nvidia/Qwen3.6-27B-NVFP4` | ASR/TTS co-residency baseline |
| `qwen-dflash` | `nvidia/Qwen3.6-27B-NVFP4` plus pinned `z-lab/Qwen3.6-27B-DFlash` | DFlash=12, GPU utilization 0.4 |
| `qwen35` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | FP8 KV, FlashInfer, MTP=3 |
| `qwen35-q4kv` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | INT4 KV, Triton attention, MTP=2 |

All current profiles use a 262,144-token context and retain up to four
concurrent sequences.

## Commands

```bash
./dgx-model list vllm
./dgx-model show vllm qwen-dflash
./dgx-model image pull vllm
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

Only `compose.yaml` and direct `profiles/*.args` files are deployable
configuration sources. Historical tuning candidates are not kept in the
workspace; backup files and directories are ignored and must never be used as
runtime configuration sources.
