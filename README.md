# Peter-Project：車牌辨識系統

用 PyTorch 從零訓練自己的車牌文字辨識模型，並提供網站展示辨識效果。

## 架構

```
generate_dataset.py ──► dataset/（合成車牌圖片 + labels.csv）
                              │
train.py ── CRNN + CTC ───────┴──► models/runs/<日期-時間>/（每次訓練各存一份）
                                          │
app.py（Flask 網站）◄─────────────────────┘
  ├ 訓練工作台：從網頁產生資料集、設定參數並訓練
  ├ 已儲存的模型：列出每次訓練結果，可切換使用
  ├ 上傳照片、框選車牌 → 辨識
  ├ 隨機產生新車牌測試
  ├ 驗證集抽樣結果（對 / 錯）
  └ 訓練曲線（訓練時會自動更新）
```

### 模型：CRNN + CTC（`plate_model.py`）

1. 影像轉灰階、縮放成 32×128
2. **CNN**（5 層卷積）抽取特徵，把高度壓成 1、寬度保留 32 → 32 個「時間步」
3. **雙向 LSTM** 從左到右、右到左讀這 32 個特徵
4. 每個時間步輸出 36 類機率（35 個字元 + CTC blank）
5. **CTC Loss** 讓模型不需要標註每個字的位置就能學會讀整串文字；推論時用 greedy decode 合併重複字並去掉 blank

### 資料集（`generate_dataset.py`）

仿台灣車牌格式合成，不需要下載任何資料：

- 格式：`ABC-1234`、`AB-1234`、`1234-AB`、`ABC-123`、`123-ABC`（不使用 I、O）
- 顏色：白底黑字、白底紅字、白底綠字、綠底白字、黃底黑字、紅底白字
- 字型：Windows 內建 Arial Narrow、Bahnschrift、Impact 等 7 種
- 增強：透視變形、隨機背景、陰影、亮度/對比、高斯模糊、動態模糊、雜訊、低解析度、JPEG 壓縮

## 使用方式

```bash
pip install -r requirements.txt

# 1. 產生資料集（預設 train 30000 張、val 3000 張；只需產生一次，已存在會自動略過）
python generate_dataset.py
python generate_dataset.py --train 50000 --val 5000 --force   # 想重新產生 / 改變數量

# 2. 訓練模型
python train.py
python train.py --epochs 50 --batch 256 --lr 2e-3 --name big-lr   # 調整參數並命名

# 3. 開啟網站（訓練時也可以開著，會自動載入最新模型）
python app.py
```

開啟 http://127.0.0.1:5000 （上面的產生資料集、訓練也都可以直接在網頁上操作）

## 模型存放位置

每次訓練都存成獨立資料夾，不會覆蓋之前的結果：

```
models/
├── active.txt                     網站目前使用哪個模型（訓練開始後自動切到新模型）
└── runs/
    ├── 20261001-230000-baseline/
    │   ├── plate_crnn.pt          驗證集最佳的模型權重
    │   ├── history.json           每個 epoch 的 loss / 正確率
    │   └── config.json            訓練參數與最佳成績
    └── 20261002-101500-big-lr/
        └── ...
```

在網頁「已儲存的模型」表格可以比較每次的成績並切換使用。不要的模型直接刪掉該資料夾即可。

## 可以嘗試的實驗

- 改 `generate_dataset.py` 的增強強度，觀察對真實照片的影響
- 改 `plate_model.py` 的網路結構（層數、通道數、LSTM 大小）
- 比較不同資料量、epoch 數、learning rate 的訓練曲線
- 用自己拍的車牌照片測試，找出模型的弱點，再回頭調整資料集

## 限制

- 只做「文字辨識」，需要先框選車牌位置；完整系統還需要車牌偵測模型（例如 YOLO）
- 資料是合成的，與真實照片仍有差距（字型、反光、髒污等）
