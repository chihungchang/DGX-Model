# SC117 IQ3_S on two DGX Sparks

Profile: `qwen38-sc117-iq3s-tp2-mtp`.

Stopped at the user's request on 2026-10-07 (Asia/Taipei) to restore the original
SGLang Huihui Qwen3.8-27B NVFP4 + DFlash2 service. The Flash profile and all three
verified model files are retained. Its saved configuration has three parallel
requests and a 262144-token context per request. The remote RPC worker is stopped
with automatic restart disabled; local Flash containers have been removed by
`dgx-model stop`. ComfyUI remains running.

SGLang was verified ready at 08:50 Asia/Taipei with TP2, DFlash2 draft length 8,
context 262144, and at most three running requests. The OpenAI chat smoke test
passed, and both NCCL NET/IB links were confirmed. See the
[switch verification](backups/switch-to-sglang-20261007-084758/verification.json).

The Dockerfile and experimental NCCL patch were recovered from the original
BuildKit cache. When activated, it uses `http://10.0.0.100:8080/v1`, the same
address used by the restored SGLang deployment.

| Setting | Value |
| --- | --- |
| Main model | `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, IQ3_S |
| Main revision | `eec4e10a2c29b440a2ae85591fa11569ce752963` |
| Context per request / parallel requests | 262144 / 3 |
| Total context allocation (`--ctx-size`) | 786432 (3 × 262144) |
| Target placement | RPC0 on spark1 + RPC1 on spark2, tensor split 1:1 |
| Draft | ggml-org Q4_0 MTP, depth 4, CUDA0 on spark1 |
| KV cache | FP16 |
| Weight residency | `--load-mode mlock --lazy-mode off`, all GPU layers |
| Network | NCCL NET/IB, `rocep1s0f0,roceP2p1s0f0` |
| API / alias | port 8080 / `qwen3.8-flash-next-sc117-iq3s` |
| Image | `dgx-llamacpp:cuda13-rpc-nccl-d7a695ef6791-v1` |
| llama.cpp source | `d7a695ef679138c13d86359b84c1731d36213d32` |

The image requires the included experimental RPC/NCCL patch; a stock upstream
image does not reproduce this configuration. All three containers use the same
image. Its build uses the existing NCCL 2.30.7 CUDA 13 artifacts at
`/home/chihung/nccl/build`; the Dockerfile verifies their SHA-256 hashes.

## Prepare

Run from `/home/chihung/ai` on spark1:

```bash
./dgx-model show llamacpp qwen38-sc117-iq3s-tp2-mtp
./dgx-model image build llamacpp qwen38-sc117-iq3s-tp2-mtp
./dgx-model model pull llamacpp qwen38-sc117-iq3s-tp2-mtp
./dgx-model validate llamacpp qwen38-sc117-iq3s-tp2-mtp
ssh spark2 'mkdir -p /home/chihung/ai/llama.cpp'
scp llama.cpp/compose.tp2.yaml spark2:/home/chihung/ai/llama.cpp/
set -o pipefail
docker image save dgx-llamacpp:cuda13-rpc-nccl-d7a695ef6791-v1 | ssh spark2 docker image load
```

The two main shards and draft total 86,063,239,136 bytes (about 80.15 GiB).
They are downloaded to spark1; spark2 receives its target tensors through RPC.
The profile records pinned download URLs and SHA-256 hashes for all three files.

## Activate when switching from SGLang

```bash
./sglang/dflash2-tp2.sh stop
ssh spark2 'cd /home/chihung/ai/llama.cpp && docker compose -f compose.tp2.yaml up -d --no-build rpc'
./dgx-model use llamacpp qwen38-sc117-iq3s-tp2-mtp
```

`dgx-model` starts the API and local RPC worker; the remote worker is started
explicitly. Stop the currently active main inference service before switching,
since both deployments use port 8080 and substantial GPU memory.

## Three-request verification, 2026-10-07

The retained profile has `--parallel 3` and `--ctx-size 786432`, retaining
262144 tokens per slot. Startup logs and `/props` confirm these limits for both
target and draft contexts. Three simultaneous OpenAI chat requests each generated
128 tokens with the correct distinct answer, occupied all three slots together,
and completed with MTP active, no truncation, and no queued requests observed.
All three inference containers remained running without restarts; both NCCL
NET/IB links were active. ComfyUI remained healthy.

Whole-host memory after the test, at 01:02:56 Asia/Taipei, with weights loaded
and the API idle (`free --bytes`; includes the OS and other services):

| Host | Total (GiB) | Used (GiB) | Available (GiB) |
| --- | ---: | ---: | ---: |
| spark1 | 121.63 | 98.04 | 23.59 |
| spark2 | 121.63 | 50.53 | 71.10 |

This was a concurrency smoke test, not three simultaneous full-256K inputs or
a new speed benchmark. Evidence: [verification](backups/parallel3-20261007-005804/verification.json),
[memory](backups/parallel3-20261007-005804/memory.json), and the raw responses,
monitor samples, active profile, and both nodes' logs in that directory.

## Same-prompt comparison, 2026-10-07 (one parallel slot)

These measurements preceded the change to three parallel slots. They describe
the original one-slot configuration, not throughput at concurrency 3.

The baseline is `edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4`
on SGLang TP2, using `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, draft
length 8, and FP8 KV. Flash uses the one-slot version of the IQ3_S / MTP4 /
FP16 KV configuration above.
This compares the two complete deployments; the models, quantization, draft
methods, and engines differ, so the difference cannot be attributed to TP2 or
the inference engine alone.

Both deployments have 256K serving capacity. Each workload uses identical input
token IDs, concurrency 1, temperature 0, thinking off, and a cold prompt cache.
Each case has one 128-token warmup and three measured 512-token generations;
the tables report medians. EOS is ignored to keep output lengths equal.
The long workload is **32768 input tokens, not a full 256K-input test**.

Decode is measured at the client from the native streaming cumulative token
counters: `(last_count - first_count) / (last_arrival - first_arrival)`.
It excludes time to first token (TTFT). SGLang uses `/generate` and llama.cpp
uses `/completion`, with the same `requests.iter_lines(chunk_size=1)` timing.

| Workload | Input tokens | SGLang decode (tok/s) | Flash decode (tok/s) | Change | Flash range (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traditional Chinese short prompt | 50 | 33.80 | 35.93 | +6.3% | 35.81–36.48 |
| English short prompt | 56 | 52.65 | 43.78 | −16.9% | 43.20–44.13 |
| Code short prompt | 61 | 69.54 | 54.75 | −21.3% | 54.74–55.92 |
| Chinese long document | 32768 | 32.62 | 37.59 | +15.2% | 36.86–39.45 |

| Workload | SGLang TTFT (s) | Flash TTFT (s) | SGLang time to last token (s) | Flash time to last token (s) |
| --- | ---: | ---: | ---: | ---: |
| Traditional Chinese short prompt | 0.106 | 0.479 | 15.22 | 14.70 |
| English short prompt | 0.110 | 0.508 | 9.82 | 12.17 |
| Code short prompt | 0.115 | 0.493 | 7.46 | 9.78 |
| Chinese long document | 11.918 | 41.154 | 27.58 | 54.97 |

Flash decodes these Chinese workloads faster, but English and code are slower.
For the 32K document, slower prompt processing outweighs faster decode: the
last output token arrives after 54.97 seconds versus 27.58 seconds on SGLang.
Full native-response receipt, including terminal metadata, takes 57.77 seconds
on Flash versus 27.58 seconds on SGLang. The llama.cpp terminal response echoes
the long prompt; that extra response-processing time is excluded from decode
and time-to-last-token above.

All 16 Flash requests completed successfully, with no truncation, no reused
prompt tokens, and nonzero MTP draft counts. Scheduler logs show exactly the
expected 16 generation tasks at context 262144, and monitoring observed no
queued competing requests. All three containers used the same image and had
no restarts or process swap. Startup logs confirm 49/49 GPU layers, locked
host PLE buffers, and NCCL `NET/IB` on both ConnectX-7 links; both links recorded
traffic during the benchmark. Short-case output text was identical across
repeats; the long case produced three variants despite temperature 0.

Evidence is retained in
[`backups/same-prompts-256k-20261007`](backups/same-prompts-256k-20261007):
[comparison JSON](backups/same-prompts-256k-20261007/comparison.json),
[CSV](backups/same-prompts-256k-20261007/comparison.csv),
[audit](backups/same-prompts-256k-20261007/audit.json),
the benchmark script, raw streams, exact inputs, profiles, and before/after logs.
The SGLang baseline is in
[`../sglang/backups/speed-restored-20261006-233704`](../sglang/backups/speed-restored-20261006-233704).
The preliminary 33792-context run is excluded from this comparison.

## Historical measurement with a different protocol

Before deletion, the final fully resident 32K test measured 40.20 tok/s decode
on TP2 and 36.98 tok/s on one Spark. Each configuration had one measured run,
temperature 0, thinking off, and 256 generated tokens. Decode used llama.cpp
`timings.predicted_per_second`. These are historical measurements, not a new
benchmark of the restored files and not directly comparable to different prompts
or client streaming timing methods.
