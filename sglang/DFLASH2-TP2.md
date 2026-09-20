# Qwen3.8-27B DFlash2 雙機 TP2

獨立部署，API `http://10.0.0.100:8080/v1`，模型名稱 `qwen3.8-27b`。
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

- Image: `dgx-sglang:qwen38-dflash2-mmcache-20260914`
- Base: 官方 ARM64 CUDA 13 nightly `lmsysorg/sglang:nightly-dev-cu13-20260914-4358a161`。
- 固定 base manifest: `sha256:c61ec2dad8d8fe040b4e341bc63ede3ff13f14598869e647ea28b4b1f36bd7d5`。
- SGLang commit `4358a1617cad734fc37bf53d9e5c2092bd43d075`，版本 `0.0.0.dev1+g4358a1617`；保留既有本機 mRoPE 修正。
- Torch `2.13.0+cu130`、Transformers `5.12.1`。
- 主模型沿用 `RadixArk/Qwen3.8-27B-NVFP4`，revision `319f741cce68d7914884900c138a1fbb70a42f30`。
- 草稿沿用 `incoai/Qwen3.8-27B-DFlash2`，revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`。
- 草稿權重 SHA256: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`，兩台一致。
- TP=2、nnodes=2、DFLASH、8 draft tokens、FlashInfer、FP8 KV、prefill chunk 8192。
- context limit 262144、mem fraction 0.50、max running requests 3、mamba extra_buffer。
- 圖片前處理快取 `--mm-preprocess-cache-size-mb 2048`；雙機特徵傳輸 `--mm-feature-transport cpu`。
- Qwen3.8 此 checkpoint 的實際 `model_type` 為 `qwen3_5`，需要明確設定正快取容量才會走新圖片 artifact 流程。
- 不開 Torch Compile，保留 decode CUDA graph（max bs 4），關閉 prefill CUDA graph。
- 通訊使用 ConnectX-7 兩個 RoCE HCA，NCCL NET=IB，rendezvous `192.168.100.10:50010`。
- ComfyUI 保留；context limit 不代表所有長度都已做容量／品質驗證。

既有單機 DFlash2 文件記錄了重疊請求可能造成上下文錯誤的上游問題，因此初始部署限制一個 active request；2026-09-12 依使用者要求調整為三個 active requests。
這不是效能調校完成的設定，也不宣稱已重現社群 76–87 tok/s。

## 驗證

```bash
python3 smoke-mm-cache.py --output-dir /tmp/sglang-mm-smoke-$(date +%Y%m%d-%H%M%S)
```

2026-09-14 升級後通過：實際 TP=2 / nnodes=2 / DFLASH / draft=8 / max-running=3 / 圖片快取 2048 MiB，Chat 與 Responses 算術串流，以及 17 張合成圖片的首輪、多輪與混合文字請求。圖片讀值與文字答案均正確；NCCL 日誌確認兩個 NET/IB 通道。

17 張圖片暖快取混合測試中，圖片 TTFT 從 26.670 秒降到 15.433 秒；圖片送出 5 秒後的 hello 從 21.669 秒降到 10.428 秒。單輪觀測，不是延遲分布。首次圖片請求則從 41.908 秒增至 55.487 秒，不能宣稱所有場景都變快。

Scheduler 顯示重複圖片命中 13056／13120 個 prefix token，但 API usage 仍回報 cached_tokens=0。剩餘取樣顯示 API／scheduler 仍有同步張量序列化、還原及跨節點廣播，升級沒有完全排除短請求等待。

詳細版本、取樣限制與測試證據見 [此次升級報告](backups/upgrade-20260914-main-cache/REPORT.md)。2026-09-09 的舊驗證保留在 `backups/dflash2-tp2-20260909/`；本次沒有驗證 215K 長對話或完整工具調用流程。

## 建置與回退

在兩台主機的 `/home/chihung/ai/sglang` 使用同一份 Dockerfile 與 patch 建置，再由第一台啟動：

```bash
docker build -f Dockerfile.dflash2-tp2 -t dgx-sglang:qwen38-dflash2-mmcache-20260914 .
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
