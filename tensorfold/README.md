# TensorFold on two DGX Sparks

Run all management commands from spark1 in /home/chihung/ai.
The retained native profile is configured to serve
`BlackwellBoy/DeepSeek-V4.1-Flash-1017-BLACKFOLD-EXL3-2.9bpw-TP2`
across both Sparks. The original LibertAIDAI pack was replaced with the pinned
BLACKFOLD overlay to fit the available disks; the original Engram tables are reused.

TensorFold was restored on both nodes on 2026-10-28 at the user's request.
The 43 backbone shards were downloaded from the pinned LibertAI base, restored
with the original BLACKFOLD overlay and checked against all author SHA-256
hashes on both nodes. Both rank caches were rebuilt, and each passed all 114
overlay-tensor slice comparisons. The retained Engram files, native runtime,
256K request limit and four-request configuration are unchanged. Text, image
token accounting, image replies, single tool calls, four simultaneous replies
and the LAN API passed after restoration.

The llama.cpp API and both RPC workers were stopped. Its Abiray GGUF backbone,
separate GGUF Engram and Q8_0 DSpark were deleted from spark1 (239.49 GiB), along
with the temporary GGUF Engram copy on spark2 (103.00 GiB). The llama.cpp
profiles, patches, images and benchmark evidence remain. Restoration receipts,
model hashes, cache provenance, API checks and deletion manifests are under
`.state/restore-20261027T230035Z/`. The previous TensorFold deletion manifests
are retained under `../llama.cpp/backups/abiray-tp2-20261027/removed-tensorfold-*`.

The served DSpark block size is the checkpoint's original five tokens, with
adaptive verification. On 2026-10-28 the user chose to keep the existing native
build and limit tuning to supported deployment parameters. The experimental
block-eight engine changes were rolled back on both nodes and in the local
build sources; the disabled patch and diagnostics remain only under
`.state/dspark8-20261028T003708Z/`.

A deployment-parameter audit on 2026-10-28 retained the original settings.
With the same greedy prose requests, the original short-prompt decode medians
were 56.48 and 56.25 tokens/s across two launches. Setting `TF_DS_SERVED_K=solo`,
`TF_DS_SOLO_OVERHEAD=1` and `TF_DS_NO_COPIES=1` together gave 55.05 tokens/s.
Setting only `TF_DS_PASS_FROM=0` gave 56.10 tokens/s; its cold 32K prefill took
14.86 s versus 15.50 and 16.44 s, and its cold 256K prefill took 119.50 s
versus 121.03 s in the final original-settings check. The earlier original
256K check took 122.28 s, so the small full-context difference is close to
baseline variation. Cached 256K TTFT also increased from 1.21 to 1.59 s.
Neither trial was retained. The 256K tests used 261,632 input and 512 output
tokens; four simultaneous short requests also passed. All measured replies
matched their reference token hashes, and the runtime, launcher and profile
hashes remained unchanged on both nodes. Requests, SSE events, memory samples
and the comparison are in `.state/params-20261028T011646Z/`.

The subsequent unchanged-native speed check sent four full 256K requests
together: 261,632 input tokens plus 512 output tokens each, text only,
temperature 0 and thinking off. Cold prefill ran one prompt at a time,
pausing the other streams' generation. First content arrived at 120.59,
241.23, 362.67 and 484.15 s; all four finished in 501.70 s. Combined cold
prefill processed 2,161.66 tokens/s. Three cached repeats each reused 260,096
tokens per request. Their median batch time was 22.28 s for 2,048 output
tokens, or 91.91 output tokens/s including cache replay and waiting.
Per-request first-to-last-content decode rate had a median of 26.82 tokens/s
(24.18 to 30.16 across the 12 cached replies). Every cached reply matched its
cold reference token hash, and monitoring confirmed four active decoding
streams. Neither rank restarted or ran out of memory; lowest sampled host
MemAvailable was 7.68/9.57 GiB. Container identities, runtime hashes and
deployment files were unchanged. Evidence: `.state/speed-4x256k-20261028T014933Z/`.

The current native deployment enables prefill interleaving at the user's
request: `TF_FILL_CHUNKS=1` and `TF_FILL_ROUNDS=4096`; `TF_FILL_MANY` is
unset, using the original disabled default. One existing prefill chunk runs
between groups of 4096 generation rounds. The original 2K/4K prefill chunk
sizes, 256K request limit, four-request capacity and DSpark block five remain
unchanged. A new prompt enters prefill after the preceding prompt finishes
prefilling. GPU time is still shared, and a chunk can briefly pause generation.
After the mixed-load trials below, the user selected this configuration again.
Only deployment parameters changed.

In the paired 32K test at 16 rounds, the generating reply's longest content gap fell
from 15.65 s to 2.14 s, with identical reply token hashes. The long prompt's
TTFT increased from 15.59 s to 23.05 s. A full-context mixed test started
three short-prompt replies, then sent a cold 261,632-token prompt with a
512-token output budget. The three existing 512-token replies finished
in 52.77-55.43 s, with maximum content gaps of 2.34 s and reply token hashes
matching their original solo runs. The 256K request's TTFT was 141.04 s;
the whole batch finished in 152.82 s. Interleaving only four rounds gave
157.54-158.04 s for the existing replies and a 153.53 s long-prompt TTFT,
so the 16-round setting was retained. Evidence:
`.state/interleave-20261028T022105Z/`. Only deployment environment variables
changed; both native binaries, kernels, launcher and model profile are retained.

The later 32-round trial reused the exact three short producer requests and
the full 256K request from the 16-round trial. Only `TF_FILL_ROUNDS` changed.
The producers' generation rates increased from 9.51-9.89 to 15.21-15.77
tokens/s, and they finished their 512-token replies in 34.33-34.63 s rather
than 52.77-55.43 s. The longest content gap increased from 2.34 to 3.02 s.
The long request's TTFT was 138.27 s versus 141.04 s, and the complete batch
took 150.27 s versus 152.82 s. All four response token hashes matched the
16-round reference. This is one mixed batch at each setting, rather than
four cold full-context prompts or a sustained-load comparison. The requested
32-round setting was retained after that trial; evidence:
`.state/interleave32-20261028T024010Z/`.

The subsequent 64-round trial again changed only `TF_FILL_ROUNDS`, with
`TF_FILL_CHUNKS=1` and all request bodies retained. The three producers
generated at 22.66-24.02 tokens/s and finished their 512-token replies in
23.09-23.33 s, versus 15.21-15.77 tokens/s and 34.33-34.63 s at 32 rounds.
The longest observed content gap was 2.16 s versus 3.02 s. The full-context
request's TTFT was 136.53 s versus 138.27 s, and the complete batch took
148.52 s versus 150.27 s. All four response token hashes matched the
32-round reference. These are single mixed-batch measurements with finite
512-token producers, rather than a sustained-load or four-full-context
comparison. The requested 64-round setting was retained after that trial; evidence:
`.state/interleave64-20261028T025100Z/`.

The requested 256-round trial reused the same requests and changed only
`TF_FILL_ROUNDS`. The three producers generated at 30.06-31.19 tokens/s,
finishing their 512-token replies in 17.78-18.29 s, versus 22.66-24.02
tokens/s and 23.09-23.33 s at 64 rounds. Their longest observed content
gap was 1.27 s versus 2.16 s. The full-context request's TTFT was 134.44 s
versus 136.53 s, and the complete batch took 146.43 s versus 148.52 s.
All four response token hashes matched the 64-round reference. In this
single mixed batch, the producers' average decode rate increased 31.5%,
while total batch time fell only 1.4%. The finite 512-token replies finish
early; these results do not establish performance under sustained
generation or four cold full-context requests. The requested 256-round
setting was retained after that trial; evidence:
`.state/interleave256-20261028T025750Z/`.

The 1024-round trial again changed only `TF_FILL_ROUNDS` and retained all
four request bodies. The three producers generated at 32.30-33.61 tokens/s,
finishing in 16.63-17.15 s, versus 30.06-31.19 tokens/s and 17.78-18.29 s
at 256 rounds. Their longest content gaps were effectively unchanged
(1.263 s versus 1.265 s). The long request's TTFT was 134.52 s versus
134.44 s, and complete batch time was 146.55 s versus 146.43 s. All four
response token hashes matched the 256-round reference. Average producer
decode improved 7.7%, with no meaningful full-batch speed-up in this single
mixed-batch comparison. These finite 512-token producers completed in
307-321 verification rounds, so the result does not establish scheduling
behavior under sustained generation. The requested 1024-round setting was
retained after that trial; evidence: `.state/interleave1024-20261028T033908Z/`.

The requested 4096-round trial changed only `TF_FILL_ROUNDS` and reused
the exact mixed requests. Producer generation was 32.38-33.69 tokens/s,
versus 32.30-33.61 tokens/s at 1024 rounds, and their 512-token replies
finished in 16.59-17.09 s versus 16.63-17.15 s. Longest content gaps were
1.275 s versus 1.263 s. The full-context request's TTFT was 136.28 s
versus 134.52 s; complete batch time was 148.34 s versus 146.55 s. All
four response token hashes matched the 1024-round reference. This single
comparison shows no meaningful extra benefit: average producer decode
changed only 0.23%, and the complete batch was 1.22% slower. Each finite
producer still completed in 307-321 verification rounds, below either
configured allowance, so longer-running generation is needed to assess
the scheduling difference. The requested 4096-round setting is retained;
evidence: `.state/interleave4096-20261028T035812Z/`.

The two-prefill/two-decoder test enabled only `TF_FILL_MANY=1`, retaining
one chunk and 4096 generation rounds. Two decoder contexts were first
preloaded; the main test reused 251,904 cached tokens in each 253,952-token
prompt and generated 8,192 tokens per decoder. Once both were streaming,
two cold 261,632-input/512-output requests were released together, 0.134 ms
apart. Every main request's full token budget was exactly 262,144.
Monitoring confirmed all four admitted with no waiting queue. The decoders
generated at 48.76/49.09 tokens/s and finished in 169.44/169.90 s; each
used only 3420/3428 verification rounds, below the 4096-round allowance.
The cold requests' first-token waits were 310.62 and 441.31 s from their
simultaneous submissions, separated by 130.68 s. This demonstrates the
strong generation preference and oldest-first tie behavior, rather than
equal interleaving of two long prefills. Main-batch wall time was 454.45 s
for 17,408 output tokens, or 38.31 tokens/s including cold prefill and
waiting; the preparatory preload time was excluded. Both cold replies
matched their cached replays. No rank restarted or ran out of memory,
and both native runtimes and the model profile remained unchanged.
Evidence: `.state/mixed-2prefill-2decode-20261028T040941Z/`.

The two-prefill/two-decoder repeat changed only `TF_FILL_ROUNDS` from 4096
to 1024, keeping `TF_FILL_CHUNKS=1`, `TF_FILL_MANY=1` and every request body
identical. The two 8,192-token decoders generated at 47.66/47.90 tokens/s
and finished in 173.36/174.07 s, versus 48.76/49.09 tokens/s and
169.44/169.90 s at 4096 rounds. The longest content gap overlapping the
cold-prefill interval was 1.57 s versus 1.25 s. Cold first-token waits were
308.57/440.89 s versus 310.62/441.31 s; complete batch time was 454.01 s
versus 454.45 s. Generation was 2.35% slower and batch time differed by
less than 0.1%, so this single comparison establishes no meaningful
overall improvement. All four complete response token hashes matched
the 4096-round reference, and both cold replies matched cached replays.
The requested 1024-round setting was retained after that trial. Native binaries, kernels,
model profile, 256K limit, four-request capacity and DSpark block five
remain unchanged. Evidence: `.state/mixed-2p2d-r1024-20261028T044459Z/`.

A matched control then tested the selected deployment unchanged:
`TF_FILL_CHUNKS=1`, `TF_FILL_ROUNDS=4096`, and `TF_FILL_MANY` unset. All four
request bodies were identical to the MANY-enabled 4096-round test. The two
8,192-token decoders generated at 48.58/49.25 tokens/s, finishing in
169.36/170.07 s. The cold first-token waits were 308.67/442.06 s, compared
with 310.62/441.31 s with MANY enabled. Complete batch time was 455.40 s
versus 454.45 s, a 0.21% difference; decoder mean rate differed by less
than 0.02%. This single matched comparison shows no meaningful speed
difference for these equal-length cold prompts. The visible scheduling
change was three admitted streams plus one waiting, rather than all four
admitted. All response token hashes matched the MANY-enabled reference,
and both cold replies matched their cached replays. No runtime or deployment
parameters changed during this test. Evidence:
`.state/mixed-2p2d-r4096-many-off-20261028T051032Z/`.

An exploratory deployment-only trial then used `TF_FILL_CHUNKS=2` and
`TF_FILL_MANY=1`, retaining 4096 rounds and the exact same two-prefill/two-decoder
requests. Mean decoder generation was 48.68 tokens/s versus 48.92 with the
selected one-chunk/MANY-off deployment. The longest decoder content gap during
cold prefill increased from 1.24 to 2.30 s. Cold first-token waits, sorted from
shortest to longest, fell from 308.67/442.06 to 298.79/431.15 s; main-batch time fell from
455.40 to 443.99 s (2.51%), excluding preparatory decoder preloads. This single
comparison shows a small latency benefit with longer generation pauses;
admitting both prefills still did not produce equal alternation. All four
complete response token hashes matched the selected baseline, and both cold
replies matched cached replays. Neither rank restarted or ran out of memory
during the trial. Afterwards, both ranks were restored to the selected
`TF_FILL_CHUNKS=1`, `TF_FILL_ROUNDS=4096`, MANY-unset deployment. Original runtime,
launcher, profile and base Compose hashes matched on both nodes, and four
concurrent reference replies passed after restoration. Evidence:
`.state/mixed-2p2d-chunks2-many1-20261028T055359Z/`.

The subsequent requested repeat retained one chunk, enabled MANY and set
1024 rounds, with every request body identical to the earlier mixed tests.
The two 8,192-token decoders generated at 47.27/47.57 tokens/s, finishing
in 174.76/175.22 s. Compared with the earlier one-chunk/MANY-on 4096-round
run, mean generation decreased 3.08% and the longest content gap increased
from 1.25 to 1.58 s. Cold first-token waits were 311.00/442.71 s versus
310.62/441.31 s. Main-batch time was 455.83 s versus 454.45 s (0.30% longer),
excluding preparatory decoder preloads. This repeat shows no speed-up for
the requested 1024-round configuration. All four full response token hashes
matched the earlier references, both cold replies matched cached replays,
and four concurrent reference replies passed. Neither rank restarted or ran
out of memory. The requested `TF_FILL_CHUNKS=1`, `TF_FILL_ROUNDS=1024`,
`TF_FILL_MANY=1` configuration was retained immediately after testing, with original
runtime, launcher and profile hashes verified and both base Compose files
matching the running containers. Evidence:
`.state/mixed-2p2d-ch1-r1024-many1-repeat-20261028T061345Z/`.

The user subsequently selected one chunk, 4096 rounds and MANY disabled again.
Both base Compose files and running ranks now use that configuration; 256K,
four-request capacity and DSpark block five are unchanged. Only the deployment
parameters changed. Both nodes passed original runtime, launcher and profile
hash checks, and four simultaneous reference replies passed with identical
token hashes. Restoration receipt:
`.state/selected-4096-many-off-restored-20261028T063011Z/`.

```bash
./dgx-model use tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model use sglang qwen-dflash2-tp2
```

Native deployment details and reproducible build: [native/README.md](native/README.md).
The profiles serve the same model ID and API address, with four simultaneous
262,144-token requests including images. The native profile's `TF_DS_WINDOW`
provides 1,245,184 shared positions independently of `--context`, including
room for native kept-prefix growth. The Python pool remains 1,114,112 positions.

```bash
./dgx-model show tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model validate tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model use tensorfold deepseek-v41-flash-1017-blackfold-exl3-native-tp2
./dgx-model status tensorfold
./dgx-model smoke tensorfold
./dgx-model logs tensorfold
```

API: http://10.0.0.100:8080/v1 (spark1's Wi-Fi DHCP address; spark2 is 10.0.0.101).
Model: `deepseek-v4.1-flash-1017`.
The Hugging Face repository ID is also accepted as an alias.
The original Python and stock-native profiles remain registered, but their deleted
LibertAI weights and rank caches must be downloaded/rebuilt before reuse. SGLang's
model files, image, Compose configuration and profiles remain available for rollback.
BLACKFOLD's generated shards are recorded in `blackfold-model-files.json`; each
node has a separate `deepseek-v41-flash-1017-blackfold` cache and a provenance receipt.
The manager validates that receipt before starting either rank.

The replacement is pinned to BLACKFOLD revision
`d795e62c91fab64531cfe6b2b5d0c0d8a95a9295`, using the author's 997,172,792-byte
overlay with SHA-256 `f42912922985684e7a0cd8f3f6d73efafec620399e8100f09cfdb0ab67b0e649`.
The official in-place converter checks all 43 output shards against the author's
hashes. Only `attn.wo_b` in layers 2–39 changes; tokenizer, vision and Engram data
are preserved. `scripts/prepare_blackfold_model.py` requires an explicit
`--replace-libertai`, stopped ranks, predownloaded authorized assets and a native
rank-cache builder. It deletes the obsolete rank cache, removes only redundant
source hardlinks, replaces one model shard at a time and rebuilds the rank cache.
Installation evidence is under `.state/blackfold-space-20261024T122138Z/`.

Python profile defaults: a 262,144-token limit per request, including text, image tokens
and its reply, and 1,114,112 tokens of shared KV storage for up to four
concurrent requests. Four requests filling their 256K windows reserve
1,056,768 pool positions after allocator alignment and scratch space.
Five DSpark/MTP drafts, text and image input, temperature 1.0.
Thinking is off by default; a request can
set reasoning_effort or chat_template_kwargs.enable_thinking.

The Python profile enables image input with --vision and --vision-urls. Send image_url
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
| Python engine | Elevatormusic/TensorFold@07aa02585f64cfb04c60a96de6554c0d4708ed46 |
| Native engine | bertholomus/TensorFold@a57deb9d470b27e1e3fab2a9494915a0049415aa |
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
