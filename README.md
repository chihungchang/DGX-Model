# DGX model management

`dgx-model` is the only management command for the main vLLM and llama.cpp
engines on this DGX Spark. Run it from `/home/chihung/ai`:

```bash
cd /home/chihung/ai
./dgx-model --help
```

The two main engines are mutually exclusive because they share DGX unified
memory. `use` validates the target profile, image, and every model artifact
before it gracefully stops the other engine and starts the selected one. Both
engines use the same fixed host/API port, `8080`.

## Common workflow

```bash
# Discover and inspect profiles without changing runtime state.
./dgx-model list
./dgx-model show vllm qwen-dflash
./dgx-model show llamacpp qwen36-fable-q4-mtp
./dgx-model validate llamacpp qwen36-fable-q4-mtp

# Prepare container images explicitly.
./dgx-model image pull vllm
./dgx-model image build llamacpp

# Download every model artifact required by a profile.
./dgx-model model pull vllm qwen-dflash
./dgx-model model pull llamacpp qwen36-fable-q4-mtp

# Discover a Hugging Face model, generate a default profile from its model
# card, pin the current revision, and download it automatically.
./dgx-model search "Qwen3.5 9B"
./dgx-model install vllm Qwen/Qwen3.5-9B
./dgx-model search "Qwen3.5 GGUF"
./dgx-model install llamacpp <org>/<repo>-GGUF

# Activate one engine. The other main engine is stopped automatically.
./dgx-model use vllm qwen-dflash
./dgx-model use llamacpp qwen36-fable-q4-mtp

# Observe and test without changing the selected model.
./dgx-model status
./dgx-model logs llamacpp
./dgx-model logs llamacpp --follow
./dgx-model smoke llamacpp

# Graceful stop is the default. With no engine, stop both main engines.
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
| `image pull vllm` | Pull the immutable vLLM image pinned by Compose digest |
| `image build llamacpp` | Build the pinned local CUDA llama.cpp image |
| `use <engine> <profile>` | Safely switch the active main engine and wait for `/v1/models` |
| `status [engine]` | Read the profile and health directly from Docker; no state file is used |
| `logs <engine> [--follow]` | Print 200 lines once, or continue streaming with `--follow` |
| `smoke <engine>` | Send and validate a small OpenAI-compatible chat completion |
| `stop [engine] [--force]` | Gracefully stop one or both engines; force is explicit |
| `profile remove ...` | Remove a profile; model files are retained by default |
| `compose-config ...` | Print raw rendered Compose configuration for debugging |

## Model downloads

For vLLM, model paths such as `/models/nvidia/Qwen3.6-27B-NVFP4` map to
`/home/chihung/models/huggingface/nvidia/Qwen3.6-27B-NVFP4`. `model pull` uses
the host `hf` CLI and downloads the complete repository snapshot, including any
drafter referenced by `--speculative-config`. Optional pinned revisions use:

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
Each result reports `vllm`, `llamacpp`, both engines, or `-` when this manager
cannot safely derive a deployment. Detection uses repository tags plus the
actual `config.json`, Safetensors/PyTorch, and GGUF files. `--limit N` accepts
1–100 results, and `--json` adds a `supported_engines` array for scripting:

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

`install` reads repository metadata and the model card without executing
any card content. For vLLM it imports supported options from the card's
standard `vllm serve` example, then applies this host's fixed address, port,
and single-GPU topology. For llama.cpp it imports supported `llama-server`
options and selects the card's recommended GGUF quantization. If none is
stated, `Q4_K_M` is preferred. Sharded GGUFs and an available multimodal
projector are downloaded as a complete set.

```bash
# The default profile name is derived from the repository name.
./dgx-model install vllm Qwen/Qwen3.5-9B

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
- `llama.cpp/compose.yaml`: shared llama.cpp container/build configuration
- `llama.cpp/profiles/*.args`: llama.cpp arguments and download metadata
- `llama.cpp/launch.sh`: small container-side llama.cpp entrypoint and artifact preflight

The container launchers parse one literal argument per line. They do not source
profiles, use `eval`, or expand environment variables.
