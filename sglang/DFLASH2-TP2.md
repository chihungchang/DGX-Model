# Qwen3.8-27B DFlash2 雙機 TP2

2026-10-16 併發上限由 3 調整為 5，兩台 profiles 同步設定
`--max-running-requests 5` 與 `--cuda-graph-max-bs-decode 5`。
這是整個雙機 TP2 服務共用的五筆上限。驗證與原三筆配置的回退資料見
[`backups/concurrent5-20261016-101413/REPORT.txt`](backups/concurrent5-20261016-101413/REPORT.txt)。
已送出五筆並行請求並觀察到 running=5，五筆答案正確；兩台均無新增 OOM、容器重啟 0。

2026-10-16 兩台配置更新為官方 ARM64 CUDA 13 nightly
`lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0`，本機映像為
`dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1`。
固定 manifest 為 `sha256:0b3e39704480c423d8a00ca15ebee63154defc62944df2d4b11c594563bab457`。
官方映像內的實際 Git HEAD 是 `c15edf314d20073630e672eb477cbcc3c58fac57`，
版本 `0.0.0.dev1+gc15edf314`，比映像標籤的提交多六個上游提交；
官方基底映像的 Git 工作樹乾淨，三個補丁目標檔案與兩個提交的上游來源雜湊均已核對。

| 本地修正 | 本次處理與證據 |
| --- | --- |
| Qwen mRoPE native 路徑繞道 | 上游 [#34446](https://github.com/sgl-project/sglang/pull/34446) 已讓 fused kernel 正確使用時間／高度／寬度位置；兩台各三個數值與形狀測試通過，本 TP2 映像不再套用舊繞道。共用補丁檔仍供較早的其他映像使用。 |
| 多圖 CPU pickle 記憶體放大 | 上游仍只對 batch slice 呼叫 `contiguous()`；原版在八張圖片的 storage 回歸測試失敗，保留 feature 與 grid 的 CPU clone 修正。兩台建置均通過 1／8／32 張、冷／暖快取與兩次 pickle 往返；32 張的 2 MiB 特徵序列化為 2.0208 MiB。 |
| Responses 原始請求及終止日誌 | 上游尚無對應掛點，保留本地補丁。10/15 與 10/16 的來源檔案相同，已列入 SHA-256 allowlist；兩台各五個 RequestLogger 回歸測試通過。 |

版本、來源比對、實際 API／多圖驗證與兩台記憶體取樣見
[`backups/upgrade-nightly-20261016-094504/REPORT.txt`](backups/upgrade-nightly-20261016-094504/REPORT.txt)。
此次先驗證 10/15 版，期間官方發布 10/16 版，因此繼續升到上述版本。
10/16 升級時實際驗證通過 Chat／Responses 完整與串流、輸出長度限制、工具呼叫與三筆並行。
64 張圖片（49,307 input tokens）的冷／暖快取皆讀值正確，暖請求回報 49,280 cached tokens；
Responses 原始 request、實際上限、engine reason 與 SSE terminal 已逐筆核對。
兩台驗證期間均無新增 OOM、容器重啟 0；可用記憶體最低為 35.34／54.27 GiB。

回退到已驗證的 10/15 版：先由第一台執行 `./dflash2-tp2.sh stop`，
兩台各從 `backups/upgrade-nightly-20261016-094504/before/spark1/` 或
`before/spark2/` 還原 `Dockerfile.dflash2-tp2`、其 `.dockerignore`、
`compose.dflash2-tp2.yaml` 與 `dflash2-tp2.sh`，再由第一台執行 `./dflash2-tp2.sh start`。
舊映像均保留；原始 9/14 部署的兩台配置另存於
`backups/upgrade-nightly-20261016-091445/before/`。

2026-10-12 加入 Responses 原始請求與終止原因紀錄，兩台當時改用
`dgx-sglang:qwen38-dflash2-observability-20261012`。只擴充既有
`RequestLogger` 的 Responses 呼叫入口及 terminal log；取樣、生成上限轉換、
停止條件、模型與 DFlash 參數保持原狀。補丁為
[`patches/patch_responses_observability.py`](patches/patch_responses_observability.py)，
CPU 檢查為 [`tests/test_responses_observability.py`](tests/test_responses_observability.py)。
部署前已確認 running／waiting／paused 皆為 0；舊映像及兩台原配置保留。

目前 profiles 開啟 `--log-requests --log-requests-level 3 --log-requests-format json`：

- `request.received.openai`：`obj.request` 保存原始 JSON；外層 `obj.request_id`
  是引擎／Responses 共用 ID。HTTP Authorization header 不在 allowlist。
- `request.received`：`rid` 與轉換後的 `sampling_params.max_new_tokens`。
- `request.finished`：相同 `rid` 的實際引擎 `meta_info.finish_reason`。
- `response.terminal`：同一 ID 的 Responses 狀態、原始上限、轉換後上限、
  引擎原因與 `incomplete_details`。診斷時須與原始 SSE terminal event 對齊，
  不得由 usage 倒推出 request budget 或停止原因。

日誌寫入既有 Docker stdout，沿用 `json-file` 的 20 MiB × 3 輪替。
Level 3 包含完整對話內容；診斷匯出需放入 0700 目錄並維持檔案 0600，
勿放入 Git 或一般報告。完整 raw capture 的留存時間取決於流量。
動態 `/configure_logging` 可調整本次程序的 logging；重新建立容器則以 profiles 為準。
本次部署、原始 request／SSE／引擎 log 與驗證結果保存於
[`backups/responses-observability-20261012/REPORT.txt`](backups/responses-observability-20261012/REPORT.txt)。
實際長請求 392.5 秒後記錄 `max_output_tokens=32768`、引擎
`max_new_tokens=32766`、`finish_reason={type:length,length:32766}`，
且原始 SSE 為 `response.incomplete`／`max_output_tokens`；四筆 log 以同一 ID
核對通過。短 Responses 正常 EOS 與 Mazi 原生容器短聊天亦通過；後者使用
Chat Completions，其實際生成 cap 為 32768，不能套用 Responses 的扣 2 行為。
這是新的受控重現；原 Solola 回合仍未還原。

要回退此次 logging，先停止兩台本推論容器，從該目錄的 `before/` 恢復兩台各自的
`compose.dflash2-tp2.yaml`、`dflash2-tp2.sh` 與 `profiles/qwen-dflash2-tp2-node*.args`，
再依雙機啟動順序啟動。舊映像
`dgx-sglang:qwen38-dflash2-mmcache-oomfix-20261007` 保留。

2026-10-07 修正多圖 CPU pickle 傳輸的記憶體放大問題，兩台改用
`dgx-sglang:qwen38-dflash2-mmcache-oomfix-20261007`。Qwen 圖片 artifact
在前處理後各自持有獨立的 CPU feature／grid storage，避免每張圖片重複序列化
整批圖片資料。保留 2048 MiB 圖片快取與所有推論參數。補丁為
[`patches/patch_qwen_mm_pickle.py`](patches/patch_qwen_mm_pickle.py)，建置時執行
[`tests/test_qwen_mm_pickle.py`](tests/test_qwen_mm_pickle.py)，驗證冷／暖快取及
tokenizer → TP0 → TP1 的兩次 pickle 往返。舊映像保留以供回退。

2026-09-29 已完成同一 Huihui 主模型的 [DSpark TP2](TP2.md) 比較，排除其他請求後 DSpark 的短提示 decode 慢約 18–25%，已依要求回復本 DFlash2 部署。操作入口為 `./dflash2-tp2.sh`；DSpark 配置保留、服務停止。[同條件 decode 比較與回切驗證](backups/huihui-dspark-20260929-150205/REPORT.md)。以下為目前 DFlash2 的配置與歷史驗證。

2026-09-29 主模型改用 `edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4`，保留目前校準 NVFP4 DFlash2 草稿與其餘推論設定。[切換與驗證紀錄](backups/huihui-main-20260929-144644/REPORT.md)。

2026-09-29 草稿改用校準 NVFP4 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`，草稿長度維持 8；兩台 profiles 與啟動檢查同步更新。主模型、256K 上限、最多 3 個請求及圖片設定不變。[切換與驗證紀錄](backups/nvfp4-draft-20260929/REPORT.md)。

2026-09-29 已恢復本部署，維持 OrcaRouter uncensored NVFP4 主模型、DFlash2 草稿長度 8、256K 上限與最多 3 個請求。文字／Responses 串流與三張圖片測試通過；[回復與磁碟清理紀錄](backups/qwen-restore-20260929/REPORT.md)。

獨立部署，API `http://10.0.0.100:8080/v1`，模型名稱 `qwen3.8-27b`。
2026-09-21 主模型切換為 `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`，DFlash2 草稿長度比較 4 後恢復為 8；保留雙機配置及 API 名稱。
兩台容器名稱均為 `dgx-sglang-dflash2-tp2`。DSpark 設定保留在原有 8080 部署。
目前記憶體配置採互斥切換，不同時啟動兩套推論服務。

## 操作

在第一台 `/home/chihung/ai/sglang` 執行：

```bash
./dflash2-tp2.sh start    # 停止 DSpark／單機，啟動雙機 DFlash2
./dflash2-tp2.sh status
./dflash2-tp2.sh dspark   # 切回原雙機 DSpark，API 8080
./dflash2-tp2.sh stop
```

也可用 `./tp2.sh start` 切回 DSpark；`./tp2.sh rollback` 切回原單機。
切換會中斷進行中的請求；等待模型載入完成再使用 API。
兩套各用獨立 Docker cache volume，只有目前選中的部署自動隨 Docker 重啟。

## 固定配置

- Image: `dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1`
- Base: 官方 ARM64 CUDA 13 nightly `lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0`。
- 固定 base manifest: `sha256:0b3e39704480c423d8a00ca15ebee63154defc62944df2d4b11c594563bab457`。
- 實際 SGLang commit `c15edf314d20073630e672eb477cbcc3c58fac57`，版本 `0.0.0.dev1+gc15edf314`；使用上游 fused mRoPE，保留多圖 CPU storage 與 Responses 日誌修正。
- Torch `2.14.1+cu130`、Transformers `5.19.0`。
- 主模型 `edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4`，revision `21565d389fe573a32c1c425e0c7ade204ddb2263`，使用 checkpoint 自附的 ModelOpt 混合精度配置與 tokenizer。
- 兩台模型目錄 `/home/chihung/models/huggingface/edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4`；所有檔案大小及 Hugging Face LFS SHA-256 已核對，來源版本記於 `.dgx-source.json`。
- 草稿 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`，revision `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1`，明確指定 `--speculative-draft-model-quantization modelopt_fp4`；NVFP4 W4A4、部分投影及 selector 保留 BF16。
- 草稿權重 1,550,153,248 bytes，SHA256: `2228b9b22e93a88d84556419c879448ab6c490ae65c4c0b166f4962190ddbf26`，兩台一致；來源版本記於模型目錄 `.dgx-source.json`。
- TP=2、nnodes=2、DFLASH、8 draft tokens、FlashInfer、FP8 KV、prefill chunk 8192。
- context limit 262144、mem fraction 0.50、max running requests 5、mamba extra_buffer。
- 圖片前處理快取 `--mm-preprocess-cache-size-mb 2048`；雙機特徵傳輸 `--mm-feature-transport cpu`。
- Qwen3.8 此 checkpoint 的實際 `model_type` 為 `qwen3_5`，需要明確設定正快取容量才會走新圖片 artifact 流程。
- 不開 Torch Compile，保留 decode CUDA graph（max bs 5），關閉 prefill CUDA graph。
- 通訊使用 ConnectX-7 兩個 RoCE HCA，NCCL NET=IB，rendezvous `192.168.100.10:50010`。
- ComfyUI 保留；context limit 不代表所有長度都已做容量／品質驗證。

既有單機 DFlash2 文件記錄了重疊請求可能造成上下文錯誤的上游問題，因此初始部署限制一個 active request；2026-09-12 依使用者要求調整為三個 active requests。
這不是效能調校完成的設定，也不宣稱已重現社群 76–87 tok/s。

## 驗證

2026-10-07 多圖修正驗證通過：

- 兩台建置時的 CPU 回歸測試通過，覆蓋 1／8／32 張圖片、冷／暖快取與兩次
  pickle 往返。32 張合計 2 MiB 特徵的序列化大小由 64.04 MiB 降為 2.02 MiB。
- 實際 Responses API 送入 64 張 1024×768 JPEG（49,307 input tokens），冷請求
  正確讀出第一張的 `100`，暖請求正確讀出最後一張的 `163`；皆收到
  `response.completed`。後續 Chat 算術請求也正確回覆 `42`。
- 冷／暖請求總時間分別為 35.00／4.25 秒；暖請求命中 49,280 個 prefix token。
  這是單次功能驗證，並非效能分布。
- 兩台 scheduler 的 CPU RSS 高水位為 11.24／11.32 GiB；每約 2.4 秒取樣的
  最低系統可用記憶體為 33.14／54.04 GiB，scheduler swap 均為 0。
  CPU RSS 不包含所有 CUDA 配置。部署後至測試結束未新增 OOM，兩台容器重啟次數均為 0。
- 兩台修正後 `qwen_vl.py` SHA-256 一致：
  `e6299de27abb5976a092b0851d333005a45d6f7848fe8353463272e5dcee5c2e`。

原始結果：`backups/mm-pickle-oomfix-20261007/smoke-64/results.json`、
`memory-summary.json`、`memory-samples.jsonl` 與兩台 build logs。

2026-09-21 已恢復草稿長度 8；相同 OrcaRouter 主模型下的 4／8 正文與 Cordatus 比較見 [草稿長度 8 重測](backups/draft8-retest-20260921/REPORT.md)。

2026-09-21 草稿長度 4 的中英文正文測試，以及同版 Cordatus C1 比較見 [草稿長度測試紀錄](backups/draft4-20260921/REPORT.md)。Cordatus 的 TPS 包含 TTFT 與預設 thinking token，與關閉 thinking 的正文 decode 測試分開報告。

2026-09-21 主模型切換的驗證、中文速度比較及原始資料見 [切換紀錄](backups/orcarouter-main-20260921/REPORT.md)。以下 2026-09-14 結果屬於原 RadixArk 主模型。

```bash
python3 smoke-mm-cache.py --output-dir /tmp/sglang-mm-smoke-$(date +%Y%m%d-%H%M%S)
```

2026-09-14 升級後通過：實際 TP=2 / nnodes=2 / DFLASH / draft=8 / max-running=3 / 圖片快取 2048 MiB，Chat 與 Responses 算術串流，以及 17 張合成圖片的首輪、多輪與混合文字請求。圖片讀值與文字答案均正確；NCCL 日誌確認兩個 NET/IB 通道。

17 張圖片暖快取混合測試中，圖片 TTFT 從 26.670 秒降到 15.433 秒；圖片送出 5 秒後的 hello 從 21.669 秒降到 10.428 秒。單輪觀測，不是延遲分布。首次圖片請求則從 41.908 秒增至 55.487 秒，不能宣稱所有場景都變快。

Scheduler 顯示重複圖片命中 13056／13120 個 prefix token，但 API usage 仍回報 cached_tokens=0。剩餘取樣顯示 API／scheduler 仍有同步張量序列化、還原及跨節點廣播，升級沒有完全排除短請求等待。

詳細版本、取樣限制與測試證據見 [此次升級報告](backups/upgrade-20260914-main-cache/REPORT.md)。2026-09-09 的舊驗證保留在 `backups/dflash2-tp2-20260909/`；本次沒有驗證 215K 長對話或完整工具調用流程。

## 建置與回退

### 多圖記憶體修正

兩台使用本目錄相同的 Dockerfile、patches 與 tests 建置：

```bash
docker build -f Dockerfile.dflash2-tp2 -t dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1 .
```

由第一台執行 `./dflash2-tp2.sh stop` 再 `./dflash2-tp2.sh start` 套用。
實際多圖串流驗證可執行：

```bash
python3 smoke-mm-pickle.py --images 64 --output-dir /tmp/sglang-mm-pickle-smoke
```

此測試呼叫 Responses API，檢查同一批 64 張 1024×768 圖片的冷／暖快取、
圖片讀值、串流結束及後續文字請求。圖片總量／解析度仍會影響正常工作記憶體；
此修正處理的是每張圖片重複攜帶整批 storage 的放大問題。

回退時先由第一台停止服務，在兩台本目錄還原部署前的配置，再由第一台啟動：

```bash
cp backups/mm-pickle-oomfix-20261007/before/sglang/compose.dflash2-tp2.yaml .
cp backups/mm-pickle-oomfix-20261007/before/sglang/dflash2-tp2.sh .
```

這會恢復舊映像 `dgx-sglang:qwen38-dflash2-mmcache-20260914`，也會恢復原本的
多圖傳輸缺陷。部署與驗證原始紀錄保留在 `backups/mm-pickle-oomfix-20261007/`。

### 回到此次切換前的 OrcaRouter 主模型

第一台執行 `./dflash2-tp2.sh stop`，接著在**兩台主機**的 `/home/chihung/ai/sglang` 執行：

```bash
cp backups/huihui-main-20260929-144644/before/qwen-dflash2-tp2-node*.args profiles/
```

最後由第一台執行 `./dflash2-tp2.sh start`。此回退保留目前校準 NVFP4 草稿、草稿長度 8 與映像；原 OrcaRouter 權重保留。

以下較早的回退段落會恢復各次備份當時的主模型與草稿組合。

### 回到原 BF16 草稿（長度維持 8）

第一台執行 `./dflash2-tp2.sh stop`，接著在**兩台主機**的 `/home/chihung/ai/sglang` 執行：

```bash
cp backups/nvfp4-draft-20260929/before/qwen-dflash2-tp2-node*.args profiles/
cp backups/nvfp4-draft-20260929/before/dflash2-tp2.sh dflash2-tp2.sh
```

最後由第一台執行 `./dflash2-tp2.sh start`。原 `incoai/Qwen3.8-27B-DFlash2` 權重保留；本次沒有更換映像。

### OrcaRouter 恢復草稿長度 8

第一台執行 `./dflash2-tp2.sh stop`，兩台各自於 `/home/chihung/ai/sglang` 執行 `cp backups/draft4-20260921/before/qwen-dflash2-tp2-node*.args profiles/`，再由第一台執行 `./dflash2-tp2.sh start`。

### 回到原 RadixArk 主模型與草稿長度 8

第一台執行 `./dflash2-tp2.sh stop`，在兩台各自的 `/home/chihung/ai/sglang` 執行：

```bash
cp backups/orcarouter-main-20260921/before/qwen-dflash2-tp2-node*.args profiles/
```

再由第一台執行 `./dflash2-tp2.sh start`。原 RadixArk 權重保留；此操作保留 9/14 SGLang 映像及 DFlash2 草稿，回復 RadixArk 主模型與草稿長度 8。

### 映像建置與 9/14 升級回退

在兩台主機的 `/home/chihung/ai/sglang` 使用同一份 Dockerfile 與 patch 建置，再由第一台啟動：

```bash
docker build -f Dockerfile.dflash2-tp2 -t dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1 .
```

兩台分別建置產生的 image ID 可能不同，固定 base digest 與套用後來源檔案雜湊已確認一致。舊映像 `dgx-sglang:qwen38-dflash2-mrope-v1` 與升級前設定都保留。

如需回到本次升級前的 DFlash2，先在第一台執行 `./dflash2-tp2.sh stop`，接著在**兩台主機**的 `/home/chihung/ai/sglang` 執行：

```bash
backup_dir=backups/upgrade-20260914-main-cache
cp "$backup_dir/compose.dflash2-tp2.yaml" compose.dflash2-tp2.yaml
cp "$backup_dir/dflash2-tp2.sh" dflash2-tp2.sh
cp "$backup_dir/qwen-dflash2-tp2-node0.args" profiles/qwen-dflash2-tp2-node0.args
cp "$backup_dir/qwen-dflash2-tp2-node1.args" profiles/qwen-dflash2-tp2-node1.args
chmod +x dflash2-tp2.sh
```

最後由第一台執行 `./dflash2-tp2.sh start`，等待 API ready。`./dflash2-tp2.sh dspark` 是切換到 DSpark，並非回到這次升級前的 DFlash2 版本。

參考：[社群雙機配方](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dual-sparks/380350)、[目前社群啟動腳本](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/start-dflash.sh)。
