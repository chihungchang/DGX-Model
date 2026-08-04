# Qwen 3.6 27B DFlash tuning — DGX Spark

Date: 2026-07-20

## Selected deployment

- Profile: `qwen-dflash`
- Target: `/models/nvidia/Qwen3.6-27B-NVFP4`
- Drafter: `/models/z-lab/Qwen3.6-27B-DFlash`
- DFlash revision: `0919688658996800f86b895034249700e9481106`
- `num_speculative_tokens`: `12`
- `max_model_len`: `262144`
- `max_num_seqs`: `4`
- `max_num_batched_tokens`: `8256`
- `gpu_memory_utilization`: `0.65`
- Mamba cache: `float32`
- Prefix caching and chunked prefill: enabled
- ASR and TTS remained co-resident and healthy throughout the final validation.

## Fixed single-stream sweep

All rows used temperature 0, 256 output tokens, three runs, identical prompts, 262,144 context, four maximum sequences, batch-token limit 8,256, and GPU memory utilization 0.65. Values are median decode tokens/second.

| Draft tokens | Prose | Coding | JSON | Reasoning |
|---:|---:|---:|---:|---:|
| 8 | 25.00 | 40.51 | 71.20 | 55.00 |
| 10 | 24.60 | 38.76 | 81.87 | 60.78 |
| 12 | 24.24 | 40.07 | 90.25 | 60.92 |
| 15 | 23.79 | 35.91 | 97.69 | 64.64 |

`n=12` was selected as the balanced agent/coding setting. `n=15` is faster only for highly deterministic structured output and regresses prose and coding.

## Baseline and final profile

| Workload | Result |
|---|---:|
| MTP=2 baseline, c=1 | 22.24 tok/s |
| MTP=2 baseline, c=4 aggregate | 70.95 tok/s |
| Final DFlash coding, c=1 | 38.74 tok/s |
| Final DFlash coding, c=4 aggregate | 101.47 tok/s |
| Final DFlash 10,164-token retrieval decode | 88.21 tok/s |
| Final DFlash 10,164-token retrieval acceptance | 85.82% |

The 10,164-token retrieval test passed twice. Cold TTFT was 10.05 seconds and the prefix-cached repeat was 2.97 seconds.

## Batch-token comparison

Raising `max_num_batched_tokens` from 8,256 to 16,384 provided no material 10K-prompt TTFT improvement, reduced c=4 coding throughput to 99.69 tok/s, and reduced the reported 262K KV concurrency from about 4.57x to 4.46x. The 8,256 setting was retained.

## Verification

- Manager status: `profile=qwen-dflash`, running, healthy.
- `/v1/models`: `qwen3.6-27b`, maximum model length 262,144.
- Engine log: `SpeculativeConfig(method='dflash', ..., num_spec_tokens=12)`.
- Tool-call parser produced a valid `get_weather` call with `{"city":"台北"}`.
- `mazi-qwen3-asr`, `mazi-qwen3-tts`, and `dgx-vllm` were all healthy.
- The Compose image is pinned to the verified digest recorded in `DFlash-deployment-revision.txt`.

Rollback without deleting the DFlash deployment:

```bash
cd ~/ai/vllm
cd /home/chihung/ai && ./dgx-model use vllm qwen-mtp2-cosvc
```
