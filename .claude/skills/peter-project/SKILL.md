---
name: peter-project
description: 操作與修改這個車牌辨識專案（Peter-Project：合成資料 → CRNN+CTC 訓練 → Flask 網站 → 教學 PDF）。在這個專案裡執行、開網站、產生資料集、訓練模型、改程式、重產教學 PDF、解釋程式碼概念、裝套件或 commit/push 之前先讀這份。
---

# Peter-Project 車牌辨識：操作手冊

## 環境（Windows）

| 項目 | 現況 | 注意 |
|---|---|---|
| Python | Anaconda 3.7.6，`C:\ProgramData\Anaconda3` | 直接用 `python` / `pip` |
| PyTorch | 1.5.0，有 CUDA GPU | 30000 張圖每個 epoch 約 36 秒 |
| Pillow | 9.5.0 | **不可升級到 10 以上**：`generate_dataset.py` 用了 `draw.textsize()`、`font.getoffset()`，Pillow 10 已移除 |
| reportlab | 4.0.9 | 只有 `docs/build_pdf.py` 需要 |
| Node.js | 24.19.0 | 只拿來 `node --check static/app.js` 檢查語法 |
| GitHub CLI | **沒有裝 `gh`** | 開 PR 要用瀏覽器打開網址 |

## 常用指令（在專案根目錄執行）

```bash
python generate_dataset.py                  # 產生 dataset/（train 30000、val 3000；已存在會略過，加 --force 重產）
python train.py --epochs 30 --name baseline # 訓練，存到 models/runs/<日期-時間>-<名稱>/
python app.py                               # 網站 http://127.0.0.1:5000
python docs/build_pdf.py [--run <run 名稱>]  # 重產 docs/車牌辨識專案教學.pdf（預設用 models/active.txt 的模型）
```

- 網站必須透過 `python app.py` 開，不能直接打開 `templates/index.html`（Flask 樣板 + 後端 API）。
- 在背景（`run_in_background`）開的伺服器最多跑約 2 小時，對話結束也會停。要長時間使用，請使用者自己在終端機執行。
- 開瀏覽器：PowerShell `Start-Process "http://127.0.0.1:5000"`。

## 檔案地圖

- `charset.py`：35 個字元（0–9、A–Z 不含 I/O、`-`）；模型輸出 36 類，index 0 是 CTC blank。
- `generate_dataset.py`：`render_plate()` 畫車牌 → `augment()` 增強 → 多行程輸出 jpg + `labels.csv`。
- `plate_model.py`：`preprocess()`（灰階 32×128、-1~1）、`CRNN`、`greedy_decode()`、`load_model()`。
- `train.py`：`Progress` 類別寫 `progress.json`（給網頁進度條）並在終端機用 `\r` 印文字進度條；每個 epoch 只在整張正確率創新高時存 `plate_crnn.pt`；第 1 個 epoch 存檔後改寫 `models/active.txt`。
- `app.py`：辨識 / 模型管理 / `/api/progress` / 用 `subprocess` 跑背景工作，輸出寫進 `logs/job.log`。
- `static/app.js`：每 2 秒輪詢 `/api/jobs` 與 `/api/progress`，每 10 秒更新曲線。
- `docs/build_pdf.py`：程式碼片段直接從原始檔擷取，改了程式後重跑就會更新。
- `docs/學習筆記.txt` / `.pdf`：使用者問過的概念整理（CRNN、`super().__init__()`、tensor 形狀、LSTM 與 CTC blank、PyTorch 訓練流程、app.py 與前端、`@app.route`、decorator）。新的解釋可以補成新章節。PDF 是從 txt 轉出來的，轉換腳本**不在 repo 裡**；txt 改了要重新產生 PDF 時，需要重寫轉換程式（縮排區塊用等寬格線排，中文佔 2 格，才能保留 txt 的對齊）。
- `.gitignore` 排除 `dataset/`、`models/`、`logs/`：模型與資料只在本機。

## 測試改動時：不要弄髒使用者的 models/

用暫存資料夾跑小規模測試，不要把測試模型寫進 `models/`（會出現在網站模型清單、還可能改掉 `active.txt`）：

```bash
python generate_dataset.py --train 600 --val 100 --out <暫存>/ds
python train.py --data <暫存>/ds --out <暫存>/models --epochs 2 --batch 32
```

要測 API 但指向暫存模型：`import app; app.RUNS_DIR = ...; app.app.test_client().get(...)`。

## 已知的坑

1. **裝套件前先停掉 `app.py`。** 伺服器載入的 `PIL/_imaging.pyd` 會被 Windows 鎖住，pip 升級到一半失敗，留下 `site-packages/~il` 這類殘骸。這次 reportlab 就是這樣把 Pillow 從 8.0.1 帶到 9.5.0。安裝前先看 pip 會不會動到 Pillow / torch。
2. **用 shell heredoc 改檔要小心 `\r`、`\n`。** 在 Git Bash 用 heredoc 跑 Python 改檔時，字串裡的 `"\r"` 被寫成真正的 CR 字元，`train.py` 就壞了。精確修改請用 Edit 工具；`train.py` 是 CRLF 換行。
3. **讀含 `\r` 進度列的 log 要用 `newline=""`。** 預設文字模式會把單獨的 `\r` 轉成換行（`app.py` 的 `job_status` 已修正）。
4. **瀏覽器快取。** `app.py` 已設定 `SEND_FILE_MAX_AGE_DEFAULT = 0`；如果使用者的畫面仍是舊版，請他按 Ctrl + F5。
5. **訓練初期整張正確率 0% 是正常的。** CTC 前 1–2 個 epoch 通常是 0%，5 個 epoch 約 88%。不要把它當成 bug。
6. **改完前端要檢查語法**：`node --check static/app.js`。

## Git 流程

- 預設分支 `main`，遠端 `https://github.com/eric40424/Peter-Project.git`。
- 改動先開分支 → commit → `git push -u origin <分支>` → 用瀏覽器打開 `https://github.com/eric40424/Peter-Project/pull/new/<分支>` 讓使用者建立並合併 PR → 合併後 `git checkout main && git pull --ff-only`，刪掉本機分支。
- 例外：更新這份 skill、`docs/` 裡的筆記這類小改動，使用者要求過直接推 `main`。程式碼改動仍走分支 + PR；不確定時先問。
- commit 訊息用繁體中文（第一行摘要 + 條列說明）。

## 回答使用者的方式

- **使用者常常看不到終端機裡的長回答。** 解釋概念、說明程式碼時，除了在對話中簡短摘要，也把完整內容寫成 UTF-8 txt（存在桌面 `C:\Users\joie\Desktop\`，不要放進專案以免弄髒 git），再用 `Start-Process notepad.exe -ArgumentList '"<路徑>"'` 打開。使用者說「解釋就好」時，就只在對話中回答，不要另外產生檔案或跑程式。
- 解釋時用專案裡的真實程式碼和**實際執行的結果**當例子（例如用訓練好的模型印出 32 個時間步的輸出、實際印出 tensor 形狀），不要編造數字。示範只在記憶體裡跑，不寫入 `models/`。
- 使用者要求「先停止」時立刻停下，並說明有沒有東西還在跑、有沒有改到檔案。
- 桌面上有使用者自己的私人檔案，只讀寫自己產生的說明檔，其他檔案不要碰。

## 寫作慣例

- 程式註解、網頁文字、對使用者的說明都用**繁體中文**。
- 沿用現有風格：短小函式、簡潔的中文註解、Flask API 回傳 `jsonify(...)`、錯誤用 `jsonify(error=...)` + 狀態碼。
- 訓練結果、行數等數字要從實際檔案讀（例如 `config.json`、`history.json`），不要憑印象寫死。
