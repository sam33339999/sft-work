# Qwen3.5-9B 台灣室內設計與工地裝修助理微調專案

本專案基於 **Qwen3.5-9B** 混合線性注意力模型，針對台灣室內設計、建築裝修、工地台日語用語及工程工序進行領域監督微調（SFT / LoRA）。本說明文件提供環境配置、資料集格式、訓練流程、權重合併與部署（含 vLLM 動態 LoRA 與獨立模型服務）的完整操作指南。

---

## 目錄
1. [專案特色與硬體需求](#專案特色與硬體需求)
2. [專案目錄結構](#專案目錄結構)
3. [訓練資料集說明](#訓練資料集說明)
4. [模型微調 (SFT / LoRA)](#模型微調-sft--lora)
5. [模型評估與效果驗證](#模型評估與效果驗證)
6. [服務化部署 (vLLM)](#服務化部署-vllm)
   - [模式 A：直接載入 LoRA（不合併權重）](#模式-a直接載入-lora不合併權重)
   - [模式 B：合併為獨立模型部署（推薦）](#模式-b合併為獨立模型部署推薦)
7. [公網穿透與 API 調用範例](#公網穿透與-api-調用範例)

---

## 專案特色與硬體需求

* **基底模型**：`Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking`
  * 架構：`Qwen3_5ForConditionalGeneration`（Gated DeltaNet 線性注意力 + 全注意力 + Vision Encoder 混合架構）
  * 精度：原生 `BF16` (bfloat16)
* **硬體環境**：
  * **GPU**：單張 NVIDIA A100-SXM4-40GB（或算力 8.0+ 具備原生 BF16 支援的 GPU）
  * **顯存需求**：LoRA 微調僅需約 22 ~ 26 GB VRAM（開啟梯度檢查點）；推理服務約 20 ~ 25 GB VRAM。
* **主要軟體依賴**：
  * PyTorch 2.11+ (CUDA 13+)
  * `transformers >= 5.17.0`
  * `peft >= 0.21.0`
  * `vllm >= 0.31.0`（需支援 Qwen3.5 系列架構）

---

## 專案目錄結構

```text
/content/sft/
├── Qwen3.5-9B-.../             # 原始基底模型權重 (BF16, ~18GB)
├── Qwen3.5-9B-FineTuned/        # [合併後] 獨立微調模型 (BF16, ~18GB)
├── output_lora/
│   └── final_adapter/           # [訓練輸出] 訓練好的 LoRA Adapter 權重 (~147MB)
├── logs/
│   ├── train_sft.log            # 微調訓練即時日誌
│   ├── vllm.log                 # vLLM 推理伺服器日誌
│   └── cloudflared.log          # Cloudflare Tunnel 穿透日誌
├── sft_train_final.jsonl        # 領域對話微調資料集 (3,267 筆)
├── train_sft.py                 # SFT LoRA 核心訓練腳本
├── merge_and_export.py          # LoRA 權重合併導出腳本
├── evaluate_samples.py          # 微調能力驗證評估腳本
├── start_service.sh             # 一鍵啟動 vLLM + Cloudflare Tunnel
└── stop_service.sh              # 一鍵關閉所有背景服務
```

---

## 訓練資料集說明

資料集檔案為 [`sft_train_final.jsonl`](sft_train_final.jsonl)，每行為 JSON 格式的多輪對話：
```json
{
  "id": "sft-0001",
  "type": "屋主溝通情境",
  "messages": [
    {"role": "system", "content": "你是一位熟悉台灣室內設計與建築工地的助理，精通工地台語、日語外來語與國語工程用語的對照，回答以台灣繁體中文為主，精準、實用。"},
    {"role": "user", "content": "我家廚房跟廁所門對門，這樣有什麼問題嗎？"},
    {"role": "assistant", "content": "使用上容易互相干擾，油煙跟濕氣也會互相影響..."}
  ]
}
```

* **樣本總數**：3,267 筆
* **長度分佈**：最小 73 tokens，中位數 155 tokens，95% 樣本小於 401 tokens，最大 582 tokens。
* **損失計算優化**：腳本內實作了嚴格的 **Assistant-only Masking**，對 System Prompt、User 提問及 Header 進行標籤遮罩（`-100`），模型僅針對助手的專業回覆計算梯度與損失。

---

## 模型微調 (SFT / LoRA)

直接執行訓練腳本即可：
```bash
python3 /content/sft/train_sft.py \
    --model_dir /content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking \
    --data_path /content/sft/sft_train_final.jsonl \
    --output_dir /content/sft/output_lora \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

* **有效批次大小**：`batch_size 4` × `grad_accum 4` = 16
* **訓練耗時**：單卡 A100-40GB 上 3 個 Epochs（共 582 步）約 48 分鐘。
* **收斂表現**：Train Loss 從初始 `4.10` 收斂至 `1.04`，Eval Loss 約 `1.98`。
* **即時日誌**：訓練進度會同步輸出至終端機與 [`/content/sft/logs/train_sft.log`](logs/train_sft.log)。

---

## 模型評估與效果驗證

訓練完成後，執行驗證腳本即可測試裝修工法、工地用語與工序順序問題：
```bash
python3 /content/sft/evaluate_samples.py \
    --model_dir /content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking \
    --adapter_dir /content/sft/output_lora/final_adapter
```

---

## 服務化部署 (vLLM)

vLLM 提供兩種方式來部署此微調模型：

### 模式 A：直接載入 LoRA（不合併權重）
> **適用情境**：想保留基底模型，同時掛載一個或多個不同的 LoRA Adapter；或不希望占用額外 18GB 硬碟空間存放合併後的模型。

啟動指令如下：
```bash
vllm serve /content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking \
    --enable-lora \
    --lora-modules sft-qwen=/content/sft/output_lora/final_adapter \
    --max-lora-rank 16 \
    --served-model-name qwen3.5-base \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

* **呼叫說明**：
  * 在發送 API 請求時，將 `model` 參數填入 **`sft-qwen`** 即可調用微調後的 LoRA 模型。
  * 將 `model` 參數填入 **`qwen3.5-base`** 則可調用原始基底模型，實現同一實例雙模型調度！

---

### 模式 B：合併為獨立模型部署（推薦）
> **適用情境**：單一用途的生產環境部署，推論吞吐量最高，無需動態注入 LoRA 運算元。

1. **合併權重**：
   ```bash
   python3 /content/sft/merge_and_export.py \
       --base_model_dir /content/sft/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking \
       --adapter_dir /content/sft/output_lora/final_adapter \
       --output_dir /content/sft/Qwen3.5-9B-FineTuned
   ```
2. **啟動 vLLM**：
   ```bash
   vllm serve /content/sft/Qwen3.5-9B-FineTuned \
       --served-model-name qwen3.5-9b \
       --host 0.0.0.0 \
       --port 8000 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.9 \
       --dtype bfloat16 \
       --enable-auto-tool-choice \
       --tool-call-parser qwen3_coder \
       --reasoning-parser qwen3 \
       --trust-remote-code
   ```

---

## Cherry Studio 軟體連線與深度思考 (Thinking) 設定指引

Cherry Studio 是一款強大的開源跨平台桌面客戶端，支援 OpenAI 相容 API、即時串流、折疊式深度思考（Chain-of-Thought）與工具調用。

### 1. 新增自定義模型供應商
* **供應商類型**：選擇 `OpenAI 相容` (OpenAI Compatible)
* **API 網址 (Base URL)**：`https://<你的Tunnel網址>/v1`（本機測試填 `http://localhost:8000/v1`）
* **API Key**：填入 `EMPTY` 或任意字元
* **模型名稱**：`qwen3.5-9b`

### 2. 深度思考 (Thinking / CoT) 與參數關鍵設定
* **最大長度 (Max Tokens)**：**強烈建議設定 4096 或 8192**（或留空不限制）。
  > ⚠️ **重要注意事項**：模型進行深度思考時，會消耗約 500 ~ 2000 個 Tokens 進行專業推導分析。若 Cherry Studio 的 Max Tokens 設為 512 或 1024，可能在思考完成前就達到長度上限被強制中斷，導致回答氣泡尚未生成！
* **深度思考折疊框**：服務端已整合 `--reasoning-parser qwen3`，當模型思考時，Cherry Studio 會即時渲染「已深度思考」下拉區塊，思考結束後無縫串流正式回答內容。
* **多輪對話支援**：系統已在 Chat Template 中內建多輪思考守護機制，多輪追問依然能維持完整思考折疊與流暢回答。
* **關閉思考 (純回答模式)**：若希望極速回覆無需思考，可在對話請求的擴展參數中傳入 `enable_thinking: false`，模型將直接輸出標準解答。

---

## 公網穿透與 API 調用範例

### 1. 啟動 Cloudflare Tunnel 穿透
若在無公網 IP 的環境（如 Google Colab），可透過 Cloudflare 快速產生公開 URL：
```bash
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```
（終端機會印出類似 `https://xxxx.trycloudflare.com` 的外部網址）

### 2. cURL 呼叫範例
```bash
curl -X POST https://<你的Tunnel網址>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [
      {
        "role": "system",
        "content": "你是一位熟悉台灣室內設計與建築工地的助理，回答以台灣繁體中文為主。"
      },
      {
        "role": "user",
        "content": "請問裝修現場的「放樣」是什麼意思？"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### 3. Python (OpenAI SDK) 呼叫範例
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<你的Tunnel網址>/v1",  # 或本機 http://localhost:8000/v1
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="qwen3.5-9b",  # 若使用模式 A 則可指定 "sft-qwen"
    messages=[
        {"role": "user", "content": "廚房跟廁所門對門，如果不能改門有什麼折衷作法？"}
    ],
    max_tokens=512,
    temperature=0.7,
)

print(response.choices[0].message.content)
```
