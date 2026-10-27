# DGX model management

`dgx-model` manages vLLM, SGLang, llama.cpp, and the TensorFold deployment
across both DGX Sparks. Run it from spark1 in the repository root:

```bash
./dgx-model --help
```

The engines are mutually exclusive because they share DGX unified
memory. `use` validates the target profile, image, and every model artifact
before it gracefully stops the other engines and starts the selected one. All
engines use the same fixed host/API port, `8080`.

## Prerequisites

This project targets NVIDIA DGX Spark. The host must provide:

- Docker Engine with the Compose plugin
- NVIDIA Container Toolkit and working Docker GPU access
- Bash, Git, `curl`, `jq`, Python 3, `rich`, `flock`, and SHA-256 tools
- Hugging Face `hf` CLI for model search, metadata, and downloads

DGX Spark ships with the NVIDIA container runtime configured. The current
[NVIDIA DGX Spark guide](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
also explains GPU validation and optional non-root Docker access. `dgx-model`
invokes Docker directly, so the current user must be able to run `docker`
without `sudo`.

## Installation

Install the host utilities on Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl jq python3 python3-rich util-linux
```

Install the `hf` CLI using the
[official standalone installer](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Clone this repository and create the host model directories:

```bash
git clone https://github.com/chihungchang/DGX-Model.git
cd DGX-Model
mkdir -p "$HOME/models/huggingface" "$HOME/models/gguf"
```

Verify the installation:

```bash
docker info >/dev/null
docker compose version
hf --version
./dgx-model --help
```

Public models require no Hugging Face login. For private or gated repositories,
authenticate once on the host; credentials are not passed into serving
containers:

```bash
hf auth login
```

## Quick start

The following example prepares and starts the included `qwen` vLLM profile.
Model downloads can be large, so confirm available disk space first.

```bash
# Inspect readiness and runtime state.
./dgx-model list
./dgx-model show vllm qwen

# Prepare the image and model artifacts.
./dgx-model image pull vllm
./dgx-model model pull vllm qwen

# Validate without starting, then activate the profile.
./dgx-model validate vllm qwen
./dgx-model use vllm qwen

# Confirm health and make a test completion.
./dgx-model status vllm
./dgx-model smoke vllm

# Stop the service gracefully.
./dgx-model stop vllm
```

While an engine is healthy, its OpenAI-compatible API is available at
`http://127.0.0.1:8080/v1`.

## Repository overview

| Backend | Host model storage | Canonical configuration |
| --- | --- | --- |
| vLLM | `$HOME/models/huggingface` | `vllm/compose.yaml` and direct `vllm/profiles/*.args` files |
| SGLang | `$HOME/models/huggingface` | `sglang/compose.yaml` and direct `sglang/profiles/*.args` files |
| llama.cpp | `$HOME/models/gguf` | `llama.cpp/Dockerfile`, `llama.cpp/compose.yaml`, and direct `llama.cpp/profiles/*.args` files |
| TensorFold | `$HOME/models/tensorfold` on both nodes | `tensorfold/compose.yaml` and `tensorfold/profiles/*.args`; [deployment and switching](tensorfold/README.md) |

Backup files and directories are ignored and must not be used as configuration
sources. Runtime state is deliberately not recorded in this document because it
becomes stale; use `./dgx-model status` for the current engine, profile, and
health.

### Included vLLM profiles

| Profile | Purpose |
| --- | --- |
| `muse-glimmer` | Meta Muse Glimmer 30B BF16 with 128K context and agentic parsers |
| `muse-glimmer-dflash` | Muse Glimmer BF16 with its official DFlash drafter, verified on DGX Spark |
| `qwen` | Qwen3.6 27B NVFP4 with MTP=2 |
| `qwen-mtp2-cosvc` | Qwen3.6 27B co-residency baseline |
| `qwen-dflash` | Qwen3.6 27B NVFP4 with DFlash |
| `qwen35` | Qwen3.6 35B A3B NVFP4 with FP8 KV and MTP=3 |
| `qwen35-q4kv` | Qwen3.6 35B A3B NVFP4 with INT4 KV and MTP=2 |

### Included SGLang profiles

| Profile | Purpose |
| --- | --- |
| `qwen` | Qwen3.8 27B mixed NVFP4/FP8 with FP8 KV cache and 256K context |
| `qwen-mtp3` | Qwen3.8 27B NVFP4 with the embedded three-token MTP head |
| `qwen-dspark` | Qwen3.8 27B NVFP4 with the pinned block-7 DSpark draft |
| `qwen-dflash2` | Qwen3.8 27B NVFP4 with pinned DFlash2 draft; concurrency capped at one for correctness |
| `qwen-dflash2-tp2` | Existing two-Spark Qwen3.8 27B NVFP4/DFlash2 deployment; retains its original per-node settings |

The backend builds locally patched derivatives of pinned SGLang images. The
DSpark profile uses the exact image manifest from its published 36.6 tok/s
benchmark, while DFlash2 retains its newer development image. Both derivatives
route multidimensional image/video mRoPE through the correct native
implementation; text requests retain the fused path. See
[`sglang/BENCHMARKS.md`](sglang/BENCHMARKS.md) for the current on-device result.

### Included llama.cpp profiles

| Profile | Purpose |
| --- | --- |
| `ds4` | DeepSeek V4 Flash 0731 UD-IQ3_XXS with DSpark and 256K context |
| `qwen36-27b-q4-mtp-c4` | Qwen3.6 27B Q4_K_M with external MTP and four parallel slots |
| `qwen36-27b-q4-mtp` | Qwen3.6 27B Q4_K_M with external MTP |
| `qwen36-27b-q4` | Qwen3.6 27B Q4_K_M without speculative decoding |
| `qwen36-fable-q4-mtp` | Fable Fusion with embedded MTP and multimodal projector |
| `qwen38-flash-next-q4` | Qwen3.8 Flash Next UD-IQ4_XS with Q8 MTP=3, lazy PLE, and multimodal projector |
| `qwen38-sc117-iq3s-tp2-mtp` | SC117 IQ3_S, NCCL TP2 and Q4_0 MTP=4; [preparation and activation](llama.cpp/TP2.md) |
| `qwythos` | Qwythos 9B with embedded MTP |

### TensorFold on two Sparks

The active `deepseek-v41-flash-1017-blackfold-exl3-native-tp2` profile uses
`BlackwellBoy/DeepSeek-V4.1-Flash-1017-BLACKFOLD-EXL3-2.9bpw-TP2`, its matching 1017
Engram tables, and the native DeepSeek TP2 TensorFold engine. It serves text and image
requests with a 262,144-token limit per request, including image tokens and
the reply, and four concurrent requests on port 8080. Shared KV storage has
1,245,184 token positions. Images may use base64 data URLs or public HTTPS URLs.
The original LibertAI weights were replaced to free disk space. SGLang remains
available for rollback. Manage the installed model and switch engines from spark1:

```bash
./dgx-model validate tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model use tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model smoke tensorfold

# Return to the existing SGLang configuration on both nodes.
./dgx-model use sglang qwen-dflash2-tp2
```

The manager validates both nodes before switching and restores the previous
SGLang service if TensorFold startup fails. Model files use about 385 GiB
per node, plus 100–110 GiB for rank caches. See
[TensorFold operations](tensorfold/README.md) for version pins, storage,
ComfyUI memory handling, and the full command set.

## Usage

```bash
# Discover and inspect profiles without changing runtime state.
./dgx-model list
./dgx-model show vllm qwen-dflash
./dgx-model show sglang qwen
./dgx-model show sglang qwen-dspark
./dgx-model show sglang qwen-dflash2
./dgx-model show llamacpp ds4
./dgx-model show llamacpp qwen36-fable-q4-mtp
./dgx-model validate llamacpp qwen36-fable-q4-mtp

# Prepare container images explicitly.
./dgx-model image pull vllm
./dgx-model image pull vllm muse-glimmer
./dgx-model image build sglang
./dgx-model image build llamacpp

# Download every model artifact required by a profile.
./dgx-model model pull vllm qwen-dflash
./dgx-model model pull sglang qwen-dflash2
./dgx-model model pull sglang qwen-dspark
./dgx-model model pull llamacpp ds4
./dgx-model model pull llamacpp qwen36-fable-q4-mtp

# Discover a Hugging Face model, generate a default profile from its model
# card, pin the current revision, and download it automatically.
./dgx-model search "Qwen3.5 9B"
./dgx-model install vllm Qwen/Qwen3.5-9B
./dgx-model install sglang Qwen/Qwen3.5-9B
./dgx-model search "Qwen3.5 GGUF"
./dgx-model install llamacpp <org>/<repo>-GGUF

# Activate one engine. The other main engines are stopped automatically.
./dgx-model use vllm qwen-dflash
./dgx-model use sglang qwen-dflash2
./dgx-model use sglang qwen-dspark
./dgx-model use llamacpp ds4
./dgx-model use llamacpp qwen36-fable-q4-mtp

# Observe and test without changing the selected model.
./dgx-model status
./dgx-model logs llamacpp
./dgx-model logs llamacpp --follow
./dgx-model smoke llamacpp

# Graceful stop is the default. With no engine, stop all main engines.
./dgx-model stop llamacpp
./dgx-model stop
```

`use` is a no-op when the requested profile is already healthy. Add
`--recreate` to deliberately replace the same container. `stop --force` uses a
zero-second Docker timeout; normal `stop` preserves the two-minute Compose grace
period for in-flight requests.

## Command meanings

| Command | Effect |
| --- | --- |
| `list [engine]` | Show profiles, all-artifact readiness, and actual Docker state |
| `show <engine> <profile>` | Explain arguments, image, paths, sources, and artifact state |
| `validate <engine> <profile>` | Validate Compose, image presence, paths, and artifacts without starting anything |
| `model pull <engine> <profile>` | Synchronize a complete Hugging Face snapshot or download every GGUF/draft/mmproj file |
| `search [query] ...` | Search Hugging Face by popularity and show all locally supported engines for every result |
| `info <model-id> ...` | Preview the Hugging Face model card, or print metadata, text, JSON, or raw Markdown |
| `install <engine> <model-id> ...` | Inspect the model card, create a revision-pinned default profile, and automatically download every selected artifact |
| `image pull vllm [profile]` | Pull the immutable published image pinned by digest |
| `image build sglang [profile]` | Build the local DFlash2 and mRoPE-correctness derivative from SGLang's pinned ARM64 image |
| `image build llamacpp` | Build the pinned local CUDA llama.cpp image |
| `use <engine> <profile>` | Safely switch the active main engine and wait for `/v1/models` |
| `status [engine]` | Read the profile and health directly from Docker; no state file is used |
| `logs <engine> [--follow]` | Print 200 lines once, or continue streaming with `--follow` |
| `smoke <engine>` | Send and validate a small OpenAI-compatible chat completion |
| `stop [engine] [--force]` | Gracefully stop one or all engines; force is explicit |
| `profile remove ...` | Remove a profile; model files are retained by default |
| `compose-config ...` | Print raw rendered Compose configuration for debugging |

## Model downloads

For vLLM and SGLang, model paths such as
`/models/RadixArk/Qwen3.8-27B-NVFP4` map beneath the shared host Hugging Face model
directory at `$HOME/models/huggingface`. The `model pull` command uses the host
`hf` CLI and downloads the complete repository snapshot, including any
configured speculative drafter. Optional pinned revisions use:

```text
# hf-revision: /models/organization/repository COMMIT_SHA
```

For llama.cpp, each referenced main GGUF, external draft, and mmproj file must
have URL and SHA-256 metadata in its profile. Downloads are resumable and are
moved into place only after checksum verification. `--force` allows a verified
replacement when an existing file is incorrect.

## Hugging Face discovery and installation

Search results default to `--sort trending_score`. Available sort fields are
`downloads`, `likes`, `trending_score`, `last_modified`, and `created_at`.
Each result reports any supported combination of `vllm`, `sglang`, and
`llamacpp`, or `-` when this manager cannot safely derive a deployment.
Detection uses repository tags plus the actual `config.json`,
Safetensors/PyTorch, and GGUF files. `--limit N` accepts 1–100 results, and
`--json` adds a `supported_engines` array for scripting:

Human-readable results use a multi-line block per model so long repository IDs
cannot shift or merge the metadata fields.

```bash
# No query lists the currently trending models.
./dgx-model search
./dgx-model search --limit 20

./dgx-model search "gemma 4" --limit 20
./dgx-model search "gemma 4" --sort trending_score --limit 20
./dgx-model search "gemma GGUF" --json
```

Inspect a search result before installing it:

```bash
./dgx-model info Qwen/Qwen3.5-9B
./dgx-model info Qwen/Qwen3.5-9B --metadata
./dgx-model info Qwen/Qwen3.5-9B --metadata --json
./dgx-model info Qwen/Qwen3.5-9B --text
./dgx-model info Qwen/Qwen3.5-9B --raw
```

The default `info` view renders the model card body for the current terminal
width, including headings, lists, quotes, links, tables, and fenced code. It
uses restrained terminal styling without full-width heading boxes. `--text`
explicitly selects the same body-only view, while `--raw` prints the complete
original Markdown including its YAML frontmatter. Metadata and JSON modes
remain machine-readable and are not styled.

`install` reads repository metadata and the model card without executing any
card content. For vLLM and SGLang it imports an allowlisted subset of options
from the respective server example, then applies this host's fixed address,
port, and single-GPU topology. For llama.cpp it imports supported
`llama-server` options and selects the card's recommended GGUF quantization. If
none is stated, `Q4_K_M` is preferred. Sharded GGUFs and an available
multimodal projector are downloaded as a complete set.

```bash
# The default profile name is derived from the repository name.
./dgx-model install vllm Qwen/Qwen3.5-9B
./dgx-model install sglang Qwen/Qwen3.5-9B

# Override the profile name or GGUF selection when needed.
./dgx-model install vllm Qwen/Qwen3.5-9B --profile qwen35-9b
./dgx-model install llamacpp org/model-GGUF --gguf model-Q5_K_M.gguf
```

The generated profile pins the current repository commit. Downloads finish
before the profile is installed. Existing profiles are never replaced unless
`--force` is explicit; a profile used by a current container cannot be
replaced. Private and gated repositories use the credentials already
configured for the `hf` CLI. Later `model pull` operations on these generated
profiles also use the `hf` CLI, so authenticated downloads continue to work.

## Profile removal

```bash
./dgx-model profile remove vllm <profile>
./dgx-model profile remove sglang <profile>
./dgx-model profile remove llamacpp <profile>
./dgx-model profile remove llamacpp <profile> --delete-files
```

The default removes only `profiles/<profile>.args`. `--delete-files` also
removes unshared model artifacts and partial downloads. Data referenced by any
other installed profile is retained. A profile used by an existing container
must be stopped before removal.

## Layout

- `dgx-model`: the single host-side lifecycle and model manager
- `scripts/dgx_model_hf.py`: safe model-card parser and default-profile generator
- `scripts/dgx_model_markdown.py`: safe terminal Markdown preview renderer
- `scripts/test_dgx_model_hf.py`: profile-generation and GGUF-shard regression tests
- `vllm/compose.yaml`: shared vLLM container configuration
- `vllm/profiles/*.args`: vLLM model and serving arguments
- `vllm/launch.sh`: small container-side vLLM entrypoint and artifact preflight
- `sglang/compose.yaml`: pinned NVIDIA SGLang container configuration
- `sglang/profiles/*.args`: SGLang model and serving arguments
- `sglang/launch.sh`: small container-side SGLang entrypoint and artifact preflight
- `llama.cpp/compose.yaml`: shared llama.cpp container/build configuration
- `llama.cpp/profiles/*.args`: llama.cpp arguments and download metadata
- `llama.cpp/launch.sh`: small container-side llama.cpp entrypoint and artifact preflight

The container launchers parse one literal argument per line. They do not source
profiles, use `eval`, or expand environment variables.
