# DGX Spark SGLang backend

2026-10-16 雙機 DFlash2 TP2 的併發上限改為 5，decode CUDA graph 批次上限同步設為 5。
設定、驗證與回退資料見 [DFlash2 TP2 部署](DFLASH2-TP2.md)。

2026-10-16 DFlash2 TP2 兩台升級為官方 CUDA 13 nightly
`nightly-dev-cu13-20261016-a93f16f0`（映像內實際 SGLang commit `c15edf314`）。
上游已修復 fused mRoPE，本 TP2 映像移除舊繞道；多圖 CPU pickle 與
Responses 日誌補丁仍須保留。固定 digest、補丁核對、驗證與回退見
[DFlash2 TP2 部署](DFLASH2-TP2.md)。

2026-10-07 已修正 DFlash2 TP2 的多圖 pickle 記憶體放大問題；兩台當時使用
`qwen38-dflash2-mmcache-oomfix-20261007` 映像，保留 2 GiB 圖片快取。
修正、建置、驗證與回退方式見 [DFlash2 TP2 部署](DFLASH2-TP2.md)。

2026-09-29 目前啟用 **Huihui + 校準 NVFP4 DFlash2 雙機 TP2**，草稿長度 8。DSpark 同條件、單請求短測慢約 18–25%，比較後已依要求切回 DFlash2；DSpark 配置保留、服務停止。操作入口為 `./sglang/dflash2-tp2.sh`，API 8080／`qwen3.8-27b`。[目前部署](DFLASH2-TP2.md)、[decode 比較與回切紀錄](backups/huihui-dspark-20260929-150205/REPORT.md)。以下為較早的切換紀錄。

2026-09-29 主模型改用 `edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4`，保留目前校準 NVFP4 DFlash2 草稿、雙機 TP=2 與 API 名稱 `qwen3.8-27b`。[切換與驗證紀錄](backups/huihui-main-20260929-144644/REPORT.md)；固定配置與回退方式見 [DFlash2 TP2 部署](DFLASH2-TP2.md)。

2026-09-29 目前雙機 DFlash2 草稿改用 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`，`modelopt_fp4`、草稿長度維持 8；兩台 `qwen-dflash2-tp2-node{0,1}` profiles 已更新。[切換與驗證紀錄](backups/nvfp4-draft-20260929/REPORT.md)。

2026-09-29 已恢復 Qwen3.8-27B uncensored 雙機 DFlash2 部署；目前操作入口為 `./sglang/dflash2-tp2.sh`，API 8080。GLM 部署與權重已移除，回復驗證見 [紀錄](backups/qwen-restore-20260929/REPORT.md)。

2026-09-21 DFlash2 TP2 主模型改用 `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`，API 名稱維持 `qwen3.8-27b`；草稿仍為 `incoai/Qwen3.8-27B-DFlash2`，比較長度 4 後恢復為 8。版本與回退方式見 [DFlash2 TP2 部署](DFLASH2-TP2.md)。

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
