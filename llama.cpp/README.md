# DGX Spark llama.cpp backend

This directory is the llama.cpp backend managed exclusively by
the repository-root `dgx-model` command. It serves an OpenAI-compatible API,
web UI, health endpoint, and Prometheus metrics on port `8080`.

The local CUDA image is
`mazi/llama.cpp-deepseek-v4:0731`, built from llama.cpp commit
`6ea215d171fd31df943bf1ac8227129f2b963160`. This build includes the dedicated
Flash 0731 chat template, reasoning-effort mapping, and structured-output
handling. GGUF files are mounted read-only from `$HOME/models/gguf`.

## Installed profiles

| Profile | Context | Speculation / purpose |
| --- | ---: | --- |
| `ds4` | 262,144 | DeepSeek V4 Flash 0731 UD-IQ3_XXS with DSpark, full CUDA offload |
| `qwen36-27b-q4` | 262,144 | Qwen3.6 27B Q4_K_M baseline |
| `qwen36-27b-q4-mtp` | 262,144 | External MTP draft, two tokens |
| `qwen36-27b-q4-mtp-c4` | 1,048,576 | External MTP draft, four parallel slots |
| `qwen36-fable-q4-mtp` | 262,144 | Fable Fusion embedded MTP plus multimodal projector |
| `qwythos` | 131,072 | Qwythos 9B embedded MTP |

## Commands

```bash
./dgx-model list llamacpp
./dgx-model show llamacpp ds4
./dgx-model show llamacpp qwen36-fable-q4-mtp
./dgx-model image build llamacpp
./dgx-model model pull llamacpp ds4
./dgx-model validate llamacpp ds4
./dgx-model use llamacpp ds4
./dgx-model model pull llamacpp qwen36-fable-q4-mtp
./dgx-model validate llamacpp qwen36-fable-q4-mtp
./dgx-model use llamacpp qwen36-fable-q4-mtp
./dgx-model smoke llamacpp
./dgx-model logs llamacpp --follow
./dgx-model stop llamacpp
```

The `ds4` deployment is about 115 GB: a 103 GB target model across four pinned
GGUF shards plus a pinned 10.9 GB DSpark draft sidecar. It uses a 256K context,
Q8_0 KV cache, and five-token DSpark speculative blocks on a
single 128 GB DGX Spark. Think High is the model default; OpenAI-compatible
clients can request `reasoning_effort=max` or disable thinking for individual
requests.

`model pull` downloads and verifies every file referenced by the profile,
including the main GGUF, external MTP draft, and mmproj. Downloads use `.part`
files for resumption and move into place only after SHA-256 verification.

`launch.sh` independently checks every referenced model file before executing
`/app/llama-server`, so direct Compose starts cannot silently omit a draft or
multimodal projector.
