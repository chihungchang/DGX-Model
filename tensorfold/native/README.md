# Native TensorFold on two DGX Sparks

Run management commands from spark1. The active profile uses BLACKFOLD EXL3
2.9 bpw, derived from the pinned LibertAIDAI 1017 pack by the author's verified
overlay. The original Engram tables are reused. Stock LibertAI weights and caches
were replaced to free disk space; their Python/native profile definitions remain.

```bash
./dgx-model use tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model status tensorfold
./dgx-model smoke tensorfold

# Return both nodes to the preserved SGLang deployment:
./dgx-model use sglang qwen-dflash2-tp2
```

The API remains `http://10.0.0.100:8080/v1`, model
`deepseek-v4.1-flash-1017`, with the BLACKFOLD Hugging Face alias. Each of four
requests can use 262,144 tokens including text, image spans and its reply.
`TF_DS_WINDOW=1245184` allocates the shared positions on both ranks. Thinking
defaults off. `max_tokens` defaults to 32,768 and is capped to the remaining
request context. Sampling defaults remain temperature 1, top-p 1 and top-k off.
The native family loads the vision tower automatically and accepts base64 image
URLs and public HTTPS image URLs. PNG decoding is native; other image formats
use its Pillow helper. Neural-network execution and scheduling are native.

Rank 0 executes `tensorfold-native`; rank 1 executes `tf-dsv41-lanes`.
The existing Docker image supplies CUDA/NCCL and image-decoding dependencies;
the Python inference entry point is not used. `compose.native.yaml` mounts the
native binaries, kernel kit and adapted template from `runtime/`. The shared
weight cache is mounted read-only. The BLACKFOLD profile uses its own model view,
manifest and cache provenance receipt; a stock rank cache cannot satisfy validation.

Both inter-node control and data use the CX7 fabric, 192.168.100.10/11 and
192.168.101.10/11, GID 3, Ethernet MTU 1500 (verbs MTU 1024). The RDMA patch
negotiates the active MTU instead of assuming 4096 bytes. SGLang and ComfyUI
remain stopped while this model is active.

| Component | Pinned source |
| --- | --- |
| Native engine | bertholomus/TensorFold@a57deb9d470b27e1e3fab2a9494915a0049415aa |
| BLACKFOLD weights | BlackwellBoy/DeepSeek-V4.1-Flash-1017-BLACKFOLD-EXL3-2.9bpw-TP2@d795e62c91fab64531cfe6b2b5d0c0d8a95a9295 |
| TP2 recipe/kit v0.7.2 | bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10@30a5bf9cad93b9355574ae65e2b89f7159350f76 |
| Compiler | Zig 0.17.0, aarch64-linux |
| Local patch | `patches/0001-dgx-rdma-and-output-ceiling.patch` |
| Image API compatibility | `patches/0002-dgx-lane-image-tokenize-and-urls.patch` |
| Full-window admission | `patches/0003-dgx-full-context-round-scratch.patch` |

The local patch ports configurable GID selection and active MTU negotiation
from the native `6366d2c` branch, preserving the newer branch's peer failure
propagation. It also retains this deployment's reply-ceiling policy and checks
real prompt overflow in the upstream server tests. The new TP2 branch supplies
independent shared storage, full nucleus sampling and prompt-cache support.
The second patch sends `/tokenize` through the same DeepSeek image-span path
as chat preparation. `TF_DS_VISION_URLS=1` enables the upstream checked public
HTTPS fetcher for that path; base64 images remain available directly.
The third patch gives a full 262,144-token request an internal extent of
266,240 positions for MTP round scratch and 4096-position alignment. Four such
extents use 1,064,960 positions; three 32,768-position growth gaps bring their
placement to 1,163,264, inside the 1,245,184 shared positions. The HTTP limit remains
262,144, including the reply; internal scratch does not reduce that limit.

To rebuild, stop TensorFold first to free RAM. The build runs in a separate
container without GPU access, checks pinned archive hashes, runs the native host
tests and compiles for ARM64/sm121. It extracts the existing model's 43 image
routing biases, generates the RoPE tables and verifies all 367 kit files against
the upstream manifest. The manager copies the result to spark2 over CX7 and
validates both nodes before a switch:

```bash
./dgx-model stop tensorfold
./dgx-model image build tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model use tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
```

`runtime/build.json` pins each installed file's size and SHA-256; runtime files
and build workspaces are local, ignored by Git. The original Python image,
profile and extension cache are retained. Returning to the stock LibertAI model
requires downloading its replaced shards and rebuilding its rank caches.

## BLACKFOLD validation (2026-10-24)

Both nodes passed all 43 author shard hashes. Each rebuilt rank cache was also
checked against all 114 attention tensor slices from the pinned overlay. Text,
PNG, nine-image history, public HTTPS JPEG, single tool calls, thinking and
top-k-disabled sampling passed through the existing LAN API.

Four distinct requests each used 261,632 input tokens including one 1344×1344
image, plus 512 output tokens: 262,144 total per request. All four completed with
zero cached tokens in the cold run, taking 516.3 seconds for the batch. Cold
prefill is sequential; first-token times were 124.4, 248.5, 373.8 and 499.1 seconds.

One repeat of the same four requests reused 260,096 cached input tokens each.
The four generated concurrently at 24.6–31.0 tokens/s per request, averaging
27.9 tokens/s; mean first-token latency was 3.03 seconds and batch time 21.97
seconds. This is a single cached batch, not a multi-run performance comparison
with the original model. Both cold and cached runs reached four active requests.
Minimum host MemAvailable across the tests was 7.35 GiB on spark1 and 9.40 GiB
on spark2; neither rank was OOM-killed or restarted. Final free disk space was
39.4 GiB and 137.1 GiB. SGLang and ComfyUI remained stopped, and both RDMA rails
used GID 3 / verbs MTU 1024. Evidence is in
`../.state/blackfold-space-20261024T122138Z/deployment-report.json`.
