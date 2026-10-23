# TensorFold on two DGX Sparks

Run all management commands from spark1 in /home/chihung/ai.
The registered profile serves LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
across both Sparks.

```bash
./dgx-model show tensorfold deepseek-v41-flash-1017-exl3-tp2
./dgx-model image build tensorfold deepseek-v41-flash-1017-exl3-tp2
./dgx-model model pull tensorfold deepseek-v41-flash-1017-exl3-tp2
./dgx-model validate tensorfold deepseek-v41-flash-1017-exl3-tp2
./dgx-model use tensorfold deepseek-v41-flash-1017-exl3-tp2
./dgx-model status tensorfold
./dgx-model smoke tensorfold
./dgx-model logs tensorfold
```

API: http://10.0.0.100:8080/v1 (spark1's Wi-Fi DHCP address; spark2 is 10.0.0.101).
Model: `deepseek-v4.1-flash-1017`.
The Hugging Face repository ID is also accepted as an alias.
Defaults: a 262,144-token limit per request, including text, image tokens
and its reply, and 1,114,112 tokens of shared KV storage for up to four
concurrent requests. Four requests filling their 256K windows reserve
1,056,768 pool positions after allocator alignment and scratch space.
Five DSpark/MTP drafts, text and image input, temperature 1.0.
Thinking is off by default; a request can
set reasoning_effort or chat_template_kwargs.enable_thinking.

Image input is enabled with --vision and --vision-urls. Send image_url
content parts with base64 data URLs or HTTP(S) URLs. The image encoder
runs on spark1, using about 0.95 GiB of CUDA allocations for the loaded
tower plus temporary encoding buffers.
TF_DS_VISION_ATTENTION=fp32 selects memory-efficient FP32 attention;
TF_DS_VISION_WARM=largest checks the maximum image grid at startup.
Image tokens count toward the same 256K context as text and output.

Inter-node inference and SSH transfers use the ConnectX-7 direct links,
192.168.100.10/11 and 192.168.101.10/11. RDMA is restricted to rocep1s0f0
and roceP2p1s0f0; NCCL's socket interface is enp1s0f0np0. The rendezvous
and TensorFold control link also bind to 192.168.100.10. Wi-Fi is not used
for communication between the model ranks.

## DSML content filtering (2026-10-19)

The image also applies `patches/dsml_single_tool.py` to the pinned CUDA
server. Upstream bypassed its DeepSeek-aware text filter when a request set
`parallel_tool_calls: false`: SSE content exposed the spaced V4.1 DSML block
although the final parser correctly returned the formal tool call. The fix
applies the existing family filter before the single-call policy and also
filters unfinished blocks from final single-call content. Formal call parsing,
arguments, the one-call limit, generation and token accounting are unchanged.
Mazi retains `parallel_tool_calls: false`; no renderer filter is required.

The manager and Compose select the `-pool-v2-dsml-v1` image and validate its
`ai.dgx.tensorfold.dsml-single-tool=v1` label on both ranks. The previous
`-pool-v2` image is retained. Tests in `tests/test_dsml_single_tool.py` exercise
the real CUDA server with a deterministic, GPU-free engine, including tiny
chunks, thinking, one-call limits, HTTP JSON/SSE and truncated envelopes.
Run them in a separate test container with pytest, never in a serving rank.
Evidence, previous configuration and rollback receipts are under
`.state/dsml-20261019/`.

## DeepSeek image history admission (2026-10-19)

The pinned DeepSeek-native `DsVision` path had its own eight-image default
and did not receive `--vision-max-images`. Nine accumulated Computer Use or
`view_image` results therefore failed before GPU admission, even with ample
context remaining. `patches/deepseek_image_context.py` removes that hidden
count default and passes an explicitly configured count through `DsApp`.
With the current profile, every valid selected image is preserved in order
and at its original prepared quality. Image tokens are accumulated during
sequential decoding; reaching the effective request context stops further
decoding with `context_length_exceeded`. The existing final admission check
still accounts for text, images and the reply, and per-image byte/pixel guards
remain. There is no client-side image eviction or model-specific Cloud filter.

Both ranks use `-pool-v2-dsml-v1-images-v1`; the manager also verifies
`ai.dgx.tensorfold.image-context=v1`. The old DSML-only image and both extension
receipts are retained under `.state/images-20261019/before/`. Restore the
previous Compose/profile/manager and each rank's own extension receipt before
restarting the old image through the manager; never switch only one rank.
Tests cover 8/9/16/64 user/tool images, ordering, explicit count configuration,
early context refusal, recovery and existing image guards. Real JSON with nine
images and SSE with twelve images returned every label in order after deploy.
The upstream revision, single-tool policy, DSML fix, context pool, model weights
and CUDA dependencies are unchanged.

## Output ceilings and context admission (2026-10-23)

`patches/output_context_ceiling.py` makes the CUDA server's explicit
`max_tokens` / `max_completion_tokens` a generation ceiling. A fitting rendered
prompt is accepted in full; the existing `prepare()` guard caps generation to
`min(requested maximum, effective context - rendered prompt tokens)`. Input is
never trimmed to reserve the maximum reply. A prompt that fills the effective
window still fails before generation. Native model and safe engine capacities
remain enforced; accounting still reports actual input/output tokens.

This corrects a Mazi request with 207,521 input tokens and a 65,535-token output
ceiling: the 262,144-token service can admit its complete prompt and generate up
to 54,623 further tokens. Mazi's separate 85% provider-usage compaction trigger,
image history, single-tool policy, model weights and context pool are unchanged.
GLM's separate family-specific admission override is outside this deployment.

Both ranks must use `-pool-v2-dsml-v1-images-v1-output-ceiling-v1`; the manager
verifies `ai.dgx.tensorfold.output-ceiling=v1` together with the existing labels.
Keep the previous image and each rank's own extension receipt before switching.
The deployment backup and evidence are in `.state/output-context-20261023/`.

`tests/test_output_context_ceiling.py` covers the incident, both output fields,
JSON/SSE, chat/completions, actual usage and true full-input refusal. When running
the pinned upstream tests in a separate GPU-free container, first execute
`tests/prepare_output_context_tests.py` there to update the three admission tests
whose expected policy changed. It preserves real input-overflow checks and the
unrelated GLM policy. Do not patch tests or run Python in a serving rank.

## Switch back to the existing SGLang deployment

```bash
./dgx-model use sglang qwen-dflash2-tp2
./dgx-model status sglang
./dgx-model smoke sglang
```

The original Compose configuration, node environment files, per-node
profiles, Huihui weights, calibrated NVFP4 DFlash2 drafter and cache volumes
remain the rollback configuration. The new qwen-dflash2-tp2.args file
registers this deployment with the manager without duplicating its settings.

Both nodes are validated before either active rank is stopped. A failed
TensorFold startup restores SGLang if it was the previous engine.
Competing model containers on both nodes have automatic restart disabled.
ComfyUI is stopped while DeepSeek uses the Sparks; switching back to SGLang
restores it if the manager stopped it.

TensorFold ranks use restart=no. After a reboot or device fault, start both
ranks through the use command. The image build command also compiles CUDA
extensions on both nodes before weights are loaded. Validation checks that
the compiled files match the image. To rebuild them separately:

```bash
python3 scripts/dgx_model_cluster.py extensions tensorfold
```

With TensorFold stopped, check exact GPU gather results on both RDMA ports,
including CUDA graph replay, without loading weights:

```bash
python3 scripts/dgx_model_cluster.py rdma-probe tensorfold
```

## Versions and storage

| Component | Pinned source |
| --- | --- |
| Engine | Elevatormusic/TensorFold@07aa02585f64cfb04c60a96de6554c0d4708ed46 |
| EXL3 weights | LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw@afc7aa0aab98ee735c0f54e0a122019af1ac08f1 |
| Engram shards 47/48 | deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 |
| NGC PyTorch 26.07 | nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c |
| Structured output | xgrammar 0.2.8, matching the pinned recipe |

The image applies patches/rdma_active_mtu.py to the pinned engine. It
negotiates the smaller active verbs MTU on both ends of each RDMA link.
The current links have Ethernet MTU 1500 and verbs MTU 1024; upstream's
fixed 4096-byte setting fails with transport retry exhaustion on this setup.

patches/shared_context.py separates TF_DS_POOL_CONTEXT (shared storage)
from --context (one request's limit). Both ranks check that their pool
capacities agree before warming up. The model's native 1,048,576-token
request limit remains enforced; aggregate KV storage can exceed it because
each request has its own positions. The current profile uses 256K per
request and a 1,114,112-token shared pool with vision enabled.
When a fresh large batch starts on an idle engine, unrelated inactive RAM
prefix entries are reclaimed first to avoid fragmented extents preventing
the fourth request from entering. Matching prefixes retain reuse, and disk
prefix entries remain available.

Prefill chunks are 1,024 tokens and TF_DS_MEM_FLOOR_GIB is 5. The latter
limits CUDA allocator growth after warm-up; host memory is also measured
during load testing because CPU buffers use additional memory.

Model and Engram files occupy about 385 GiB on each node, with another
100–110 GiB for the first rank cache. Engram tables are read from disk.
Both nodes use local storage. The head's sibling .src staging directory
shares hard links with its model view rather than duplicating the weights.

- Model view: ~/models/tensorfold/deepseek-v41-flash-1017-exl3-2.9bpw/{model,engram}
- Cache: ~/.cache/tensorfold/deepseek-v41-flash-1017
- Persistent prompt-cache budget: 8 GiB per node.

Completed model shards are reused; Engram downloads resume by byte range.
Downloads check the upstream SHA-256 manifest on both nodes.
The chat template is pinned, patched and checked by hash. model-files.json
records expected sizes, revisions and source hashes.
Before switching, validation also reserves space for the first rank cache,
the prompt-cache budget, and 10 GiB of free disk space.

Containers prohibit swap and cap memory at 112 GiB. The launcher disables
transparent huge pages for the inference process through prctl. Host-wide
swap, sysctl and display-manager settings are unchanged. Never launch Python
or build extensions with docker exec inside a serving TensorFold rank.
Use its HTTP API and host-side status/log commands.

Deployment checks passed on both Sparks: exact eager and captured RDMA
gathers, a chat smoke request, all five upstream feature checks (plain,
thinking, strict JSON schema, forced tool, regex), the full model ID alias,
four simultaneous replies, and a 9,043-token prompt.

The current 256K image profile passed small and large base64 image input,
HTTP URL image input, one cold full-context request, and four simultaneous
cold full-context requests. Each long request used 261,632 input tokens,
including one 1344x1344 image encoded as 994 image tokens, and generated
512 output tokens. All four contexts were admitted; cached replay confirmed
four streams decoding together with matching reply token hashes. Both ranks
stayed running without OOM or restarts. The lowest sampled MemAvailable
across loading, preparation and testing was 3.37 GiB on spark1 and 5.85 GiB
on spark2. Cold prefill still processes prompts in order. Results:
tensorfold/.state/vision-4x256k-20261019T073046Z/.

The 2026-10-19 speed audit preserves that four-request image capacity.
It compares the unchanged configuration with shorter speculative windows,
the `even` depth policy, shared-expert overlap, and CX7 MTU 9000. Each trial
reuses the same four prompts with 261,120 cached tokens, processes the last
512 input tokens, and generates 512 tokens per request at temperature 0.
The first replay after each restart restores the disk prefixes and is
excluded. Health samples must show four streams decoding together, and
every reply's token hash must match its original cold-run reference.
Compare repeated generation rates together with whole-batch completion
time; a higher first-token delay alone can inflate the reported decode rate.
This audit measures generation with the full contexts resident. It does
not remeasure cold full-context prefill. Raw requests, results and the
comparison report are under
`tensorfold/.state/optimize-4x256k-20261019T082312Z/`.

No tuning change was retained: the alternatives did not improve both
generation rate and batch completion time consistently. The original
Compose file is restored, including the default `conf` policy and depth
table `5,5,3,3`, shared-expert overlap off, and CX7 MTU 1500 on both ports
of both Sparks. Wi-Fi configuration is unchanged. The initial unchanged
baseline's three-run median was 26.38 tokens/s per stream and 22.41 s per
batch. The final five-run check of the restored baseline measured 26.62
tokens/s per stream and 22.16 s per batch. The best-looking shorter-window
trial measured 26.72 tokens/s but took 22.82 s per batch; shared-expert
overlap measured 26.29 tokens/s and 22.45 s across five runs. These small
differences do not establish a speed-up. SGLang and ComfyUI remain stopped.

The earlier 128K image profile passed small and large image recognition through
base64 input, plus HTTP URL image input. One cold request and four
simultaneous cold requests each used 130,560 input tokens, including one
1344x1344 image encoded as 994 image tokens, and generated 512 output
tokens. All four contexts were admitted, and the cached replay confirmed
four streams decoding together with reply token hashes matching the cold
run. Both ranks stayed running without OOM or restarts. The lowest sampled
MemAvailable across loading and testing was 3.45 GiB on spark1 and 5.98 GiB
on spark2. This validates one image per request; more images and broader
visual quality need separate checks. Cold prefill still processes prompts
in order. Results: tensorfold/.state/vision-4x128k-20261019T070105Z/.

The earlier larger shared pool passed one cold 262,144-token request and four simultaneous
cold 262,144-token requests, each generating 512 tokens. All four contexts were
admitted without waiting; cold prefill still processes prompts in order,
so their first replies arrive in sequence. Reusing the four prompts also
confirmed four streams decoding together and matching the cold reply token
hashes. These are capacity and speed checks; full-window answer quality
and a single 1,048,576-token request need separate validation. Those 256K
measurements used the earlier text-only profile and larger shared pool.

The 256K tests used temperature 0, thinking off, and five MTP drafts. The
lowest sampled host MemAvailable in that earlier run was 2.22 GiB on
spark1 and 6.38 GiB on spark2 with SGLang and ComfyUI stopped. Keep those
services stopped while using this profile. Test the current 256K image profile
with 261,632 input tokens (including images) and 512 output tokens:

```bash
python3 scripts/bench_tensorfold_long_context.py \
  --output /tmp/tensorfold-bench-vision-256k --input-tokens 261632 \
  --output-tokens 512 --single-repeats 1 --worker spark2 \
  --image /path/to/test-image.png
```

Request bodies, streamed events, both nodes' memory samples and results
are saved under the output directory. Copy pg23950.txt and source.json
from a previous run into the output directory before benchmarking. The
earlier text-only run is recorded in
tensorfold/.state/context-4x256k-20261019T061447Z/.

## Upstream recipe

The pinned utilities under vendor/deepseek-v41 come from
[Elevatormusic's two-Spark recipe](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark).
UPSTREAM.json, LICENSE and CREDITS.md retain their provenance and licence.
The recipe has text, tool-call and structured-output measurements; broader
image quality and full-window answer quality need separate validation.
