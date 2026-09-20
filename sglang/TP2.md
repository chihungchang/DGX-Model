# Qwen3.8-27B DSpark 雙機張量並行

2026-09-09 最新狀態：已新增並啟用 [DFlash2 TP2](DFLASH2-TP2.md)，API 8080。本文 DSpark TP2 設定保留、目前停止；`./tp2.sh start` 可切回。以下啟用紀錄為歷史狀態。

目前狀態（2026-09-09 再次啟用）：兩台 `dgx-sglang-tp2` 運行，TP=2 / nnodes=2 / DSPARK；原單機停止且 restart=no。API 實際回覆測試通過，證據在 `backups/tp2-reenable-20260909/`。

兩台各一張 GB10 GPU，`--tp-size 2 --nnodes 2`。主模型與 DSpark 草稿沿用原單機部署的同一份權重、Docker 映像與 mRoPE 修正。

| 節點 | rank | 高速 IP | 容器 |
| --- | --- | --- | --- |
| spark1 / dgx-spark | 0 | 192.168.100.10 | dgx-sglang-tp2 |
| spark2 / dgx-spark-2 | 1 | 192.168.100.11 | dgx-sglang-tp2 |

主節點 rendezvous：`192.168.100.10:50000`。
API：`http://10.0.0.100:8080/v1`，model：`qwen3.8-27b`。
Context 設定：262144 tokens；已完成 261886 input + 256 output 的三輪零快取測試。

容器使用 host network、host IPC、明確映射 RDMA devices，NCCL 強制 IB/RoCE，使用接線中的兩個 CX7 介面。Bootstrap 與 Gloo 使用高速介面 enp1s0f0np0。
本機同時有 ComfyUI，因此設定 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`，允許兩台可用記憶體不同；啟動時共同依較少的可用記憶體配置一致的 token pool。`mem-fraction-static=0.50`。

每台保留獨立的 `dgx-sglang-tp2-cache` 卷。兩台均設 restart=unless-stopped。

## 管理

在第一台執行：

```bash
cd /home/chihung/ai/sglang
./tp2.sh start
./tp2.sh status
docker logs -f dgx-sglang-tp2
ssh spark2 docker logs --tail 100 dgx-sglang-tp2
./tp2.sh stop
```

`tp2.sh start` 會保留原容器 `dgx-sglang` 並停用其自動重啟，避免與雙機服務搶用 8080 / GPU。
回復原本單機 DSpark：

```bash
./tp2.sh rollback
```

雙機運行期間，使用 `tp2.sh` 管理；原有 `dgx-model use sglang ...` 是單機管理流程，不會協調第二台。要切回單機請先 rollback。

模型同步採用 rsync -aL（解參照 Hugging Face 快取符號連結），第二台不依賴第一台的磁碟掛載。
原部署備份：`backups/tp2-20260908/`。

## 驗證結果（2026-09-08）

- 兩台映像 ID 一致：`sha256:7ccdb535df16f8dd61fbe09c89e6b6569414a5f9e638eafd0d688b82b2d7b370`。
- `/get_server_info` 回報 `tp_size=2`、`nnodes=2`、`node_rank=0`、`speculative_algorithm=DSPARK`。
- 啟動日誌確認 TP0 / TP1；主模型權重每台約 11–12 GB，兩台皆載入 DSpark。
- SGLang 通訊日誌使用 NCCL 2.30.7+cuda13.3、NET/IB，兩個 RoCE 介面皆被選用。
- 兩台成功建立 target/draft CUDA graphs，沒有容器自動重啟。
- Wi-Fi API 位址 `/v1/models` 正常回應。
- 計算測試 `17+25` 正確回答 `42`，耗時 0.48 秒。
- 中文生成測試正常輸出 192 tokens，耗時 6.18 秒（含 prefill 與 HTTP，約 31.1 output tokens/s；因 max_tokens=192 而截斷，非正式性能 benchmark）。
- 256K 長文已於 2026-09-08/09 補測；重開機恢復或故障轉移未驗證。

證據位於 `backups/tp2-20260908/`：`server-info.json`、`smoke-results.json`、`node0-startup.log`、`node1-startup.log`。

歷史紀錄：2026-09-09 完成雙機／單機比較後，曾切回單機 `dgx-sglang`，restart policy 為 `unless-stopped`；兩台 TP2 容器已移除，配置與 cache volume 保留。單機記憶體比例調為 0.51 以容納 256K 測試。完整結果見 [比較報告](benchmarks/TP1-vs-TP2-20260908.md)。
