# DGX Spark llama.cpp backend

This directory is the llama.cpp backend managed exclusively by
the repository-root `dgx-model` command. It serves an OpenAI-compatible API,
web UI, health endpoint, and Prometheus metrics on port `8080`.

The local CUDA image is `mazi/llama.cpp:b10236-cuda13-openssl`, built from
llama.cpp commit `1464c62d88f699ec9700c8010bbfdbc603a9efd6`. GGUF files are
mounted read-only from `$HOME/models/gguf`.

## Installed profiles

| Profile | Context | Speculation / purpose |
| --- | ---: | --- |
| `qwen36-27b-q4` | 262,144 | Qwen3.6 27B Q4_K_M baseline |
| `qwen36-27b-q4-mtp` | 262,144 | External MTP draft, two tokens |
| `qwen36-27b-q4-mtp-c4` | 1,048,576 | External MTP draft, four parallel slots |
| `qwen36-fable-q4-mtp` | 262,144 | Fable Fusion embedded MTP plus multimodal projector |
| `qwythos` | 131,072 | Qwythos 9B embedded MTP |

## Commands

```bash
./dgx-model list llamacpp
./dgx-model show llamacpp qwen36-fable-q4-mtp
./dgx-model image build llamacpp
./dgx-model model pull llamacpp qwen36-fable-q4-mtp
./dgx-model validate llamacpp qwen36-fable-q4-mtp
./dgx-model use llamacpp qwen36-fable-q4-mtp
./dgx-model smoke llamacpp
./dgx-model logs llamacpp --follow
./dgx-model stop llamacpp
```

`model pull` downloads and verifies every file referenced by the profile,
including the main GGUF, external MTP draft, and mmproj. Downloads use `.part`
files for resumption and move into place only after SHA-256 verification.

`launch.sh` independently checks every referenced model file before executing
`/app/llama-server`, so direct Compose starts cannot silently omit a draft or
multimodal projector.
