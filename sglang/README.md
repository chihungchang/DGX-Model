# DGX Spark SGLang backend

2026-09-09 已完成 [單機／雙機效能比較](benchmarks/TP1-vs-TP2-20260908.md)。
2026-09-09 新增並啟用 [DFlash2 TP2 部署](DFLASH2-TP2.md)：API 8080，模型 `qwen3.8-27b`，草稿長度 8。
2026-09-14 DFlash2 雙機已升級至官方 CUDA 13 nightly `4358a161`，啟用 2048 MiB 圖片前處理快取；[實測結果與回退方式](DFLASH2-TP2.md)。以下單機／DSpark 的版本敘述保留各自部署設定。
原 DSpark TP2（8080）保留並停止，可用 `./dflash2-tp2.sh dspark` 切回；兩套採互斥啟動。
原單機容器已停止；其 0.51 記憶體設定保留供日後切回。
本次 Cordatus 單機／雙機平均為 41.76／64.99 tok/s；以下 42.85 為先前測試紀錄。

See [BENCHMARKS.md](BENCHMARKS.md) for the measured **42.85 tok/s** C1 result
across 30 requests, raw evidence, and repeatable benchmark commands. After the
firmware update and reboot, the previous approximately 721 MHz behavior was
absent: measured GPU clocks were 2431-2476 MHz during inference.

This backend is managed through the repository-root `dgx-model` command and
serves an OpenAI-compatible API on port 8080. It shares
`$HOME/models/huggingface` with vLLM, so an existing checkpoint does not need
to be downloaded again.

The default local image is built from SGLang's Qwen3.8 DFlash2 development
image. The DSpark profile instead pins the exact multi-architecture manifest
used for the published 36.6 tok/s result; on DGX Spark it resolves to the
`linux/arm64` image built from SGLang commit `c4271c3f`. Both receive the same
small mRoPE correctness patch during the local derivative build.

The derivative adds the conservative form of the Qwen3.5-family multimodal
mRoPE correctness fix: text requests retain the fused CUDA path, while image
and video requests with three-axis positions use SGLang's existing native
mRoPE implementation. This avoids silently dropping the height and width
positions in SGLang 0.5.17. The upstream optimized-kernel fix was merged as
[SGLang PR #34446](https://github.com/sgl-project/sglang/pull/34446).

## 雙機張量並行

2026-09-08 新增 Qwen3.8-27B + DSpark 的兩台 Spark TP=2 部署，使用獨立的
`compose.tp2.yaml`、`tp2.sh` 與 `dgx-sglang-tp2` 容器。
操作方式與單機回復流程見 [TP2.md](TP2.md)。

## Included profile

| Profile | Checkpoint | Important settings |
| --- | --- | --- |
| `qwen` | `RadixArk/Qwen3.8-27B-NVFP4` | ModelOpt mixed NVFP4/FP8, FP8 KV cache, 256K context |
| `qwen-mtp3` | Target above with its in-checkpoint MTP head | Three predicted tokens per MTP cycle, one concurrent request |
| `qwen-dspark` | Target above plus `RadixArk/Qwen3.8-27B-DSpark` | DSpark block 7; tuned to reproduce the 36.6 tok/s single-Spark benchmark |
| `qwen-dflash2` | Target above plus `incoai/Qwen3.8-27B-DFlash2` | DFlash2 block size 4, one concurrent request for correctness safety |

The profile follows [SGLang's verified DGX Spark
recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B):
FlashInfer attention, a 2,048-token prefill chunk, and
`--mem-fraction-static 0.80` to leave enough unified memory for DGX OS.

The MTP profile uses the checkpoint's embedded MTP head through SGLang's
EAGLE path. Three prediction steps with `topk=1` form a four-position verify
tree, expressed as `--speculative-num-steps 3` and
`--speculative-num-draft-tokens 4`.

The DSpark profile follows the published single-DGX-Spark 36.6 output tok/s
operating point: an unquantized block-7 RadixArk draft, originally a 0.50 static memory
fraction (now 0.51 to fit the 256K test), 8,192-token prefill chunks, decode CUDA graphs through batch four,
and Torch compilation through batch four. The public result used 128 input and
128 output tokens at concurrency one; longer generations are expected to vary
with draft acceptance.

The container is pinned to the GB10 performance cores (`5-9,15-19`), keeps
SGLang, TorchInductor, and Triton caches in the persistent
`dgx-sglang-cache` volume, and uses `/v1/models` for health checks. The latter
avoids the one-token generation performed by SGLang's `/health` endpoint every
health interval. `--sleep-on-idle` prevents the scheduler from busy-spinning on
one CPU core between requests.

The DFlash2 profile uses the recipe's separate 2B-parameter draft checkpoint
with `--speculative-algorithm DFLASH` and four draft tokens. The target and
draft revisions are pinned. Because upstream issue
[SGLang #36548](https://github.com/sgl-project/sglang/issues/36548) reports
possible context corruption when requests overlap, this profile sets
`--max-running-requests 1`. Use the plain `qwen` profile when concurrency is
required.

## Commands

Run these from the repository root:

```bash
./dgx-model list sglang
./dgx-model show sglang qwen
./dgx-model image build sglang
./dgx-model model pull sglang qwen
./dgx-model model pull sglang qwen-dspark
./dgx-model model pull sglang qwen-dflash2
./dgx-model validate sglang qwen
./dgx-model validate sglang qwen-dspark
./dgx-model validate sglang qwen-dflash2
./dgx-model use sglang qwen-dspark
./dgx-model use sglang qwen-dflash2
./dgx-model status sglang
./dgx-model smoke sglang
./dgx-model logs sglang --follow
./dgx-model stop sglang
```

`use` stops vLLM and llama.cpp first when either has a container that can hold
unified memory. The launcher reads one literal argument per profile line,
checks every referenced model directory, and then executes
`python3 -m sglang.launch_server`.
