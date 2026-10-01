"""訓練車牌辨識模型。

用法：
    python train.py                       # 使用 dataset/，訓練 30 個 epoch
    python train.py --epochs 50 --batch 256 --lr 2e-3

每次訓練都會存到獨立資料夾，不會覆蓋之前的模型：
    models/runs/<日期-時間>[-名稱]/
        plate_crnn.pt   驗證集最佳的模型
        history.json    每個 epoch 的 loss / 準確率（網站上會畫成曲線）
        config.json     訓練參數與最佳結果
    models/active.txt   網站目前使用的模型（訓練開始後自動切到這次的模型）
"""
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image

from plate_model import CRNN, encode, greedy_decode, preprocess


class Progress:
    """把訓練進度寫進 progress.json，讓網站畫進度條（最多每秒寫一次）。"""

    def __init__(self, path):
        self.path, self.last, self.state, self.inline = path, 0.0, {}, False

    def update(self, force=False, **fields):
        self.state.update(fields, updated=time.time())
        if not force and time.time() - self.last < 1:
            return
        self.last = time.time()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.path)
        except OSError:  # Windows 上網站剛好在讀檔時可能失敗，下次再寫即可
            pass
        self.print_bar()

    def print_bar(self):
        """在終端機（和網頁的執行紀錄）用 \r 覆寫同一行顯示進度條。"""
        st = self.state
        bar = lambda frac, n=20: "#" * int(frac * n) + "-" * (n - int(frac * n))
        if st["stage"] == "loading" and "total" in st:
            line = f"  載入 {st['split']} [{bar(st['done'] / st['total'])}] {st['done']}/{st['total']}"
        elif st["stage"] == "training":
            frac = st["batch"] / st["batches"]
            line = (f"  Epoch {st['epoch']}/{st['epochs']} [{bar(frac)}] {frac:4.0%}  "
                    f"batch {st['batch']}/{st['batches']}  預估剩餘 {clock(st['eta'])}")
        elif st["stage"] == "evaluating":
            line = f"  Epoch {st['epoch']}/{st['epochs']} 驗證中…"
        else:
            return
        print("\r" + line.ljust(70), end="", flush=True)
        self.inline = True

    def say(self, msg):
        """一般訊息：先清掉進度條那一行再印。"""
        if self.inline:
            print("\r" + " " * 70 + "\r", end="")
            self.inline = False
        print(msg, flush=True)


def clock(sec):
    m, s = divmod(int(sec), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m:02d}:{s:02d}"


def load_split(folder, progress):
    with open(os.path.join(folder, "labels.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    progress.say(f"載入 {folder}（{len(rows)} 張）…")
    split = os.path.basename(folder)
    images = []
    for i, r in enumerate(rows):
        images.append(preprocess(Image.open(os.path.join(folder, r["filename"]))))
        progress.update(stage="loading", split=split, done=i + 1, total=len(rows))
    return torch.stack(images), [r["text"] for r in rows]


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return dp[-1]


def evaluate(model, X, texts, device, batch=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            preds += [t for t, _ in greedy_decode(model(X[i:i + batch].to(device)))]
    seq_acc = sum(p == t for p, t in zip(preds, texts)) / len(texts)
    errors = sum(edit_distance(p, t) for p, t in zip(preds, texts))
    char_acc = 1 - errors / sum(len(t) for t in texts)
    return seq_acc, char_acc


def train_augment(x):
    """訓練時的即時資料增強：隨機亮度/對比與小幅平移。"""
    n = x.size(0)
    x = x * torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.7, 1.3) \
        + torch.empty(n, 1, 1, 1, device=x.device).uniform_(-0.2, 0.2)
    shift = int(torch.randint(-3, 4, (1,)))
    return torch.roll(x, shifts=shift, dims=3).clamp(-1, 1)


def main():
    p = argparse.ArgumentParser(description="訓練車牌辨識 CRNN")
    p.add_argument("--data", default="dataset")
    p.add_argument("--out", default="models")
    p.add_argument("--name", default="", help="這次訓練的名稱（會加在資料夾名稱後面）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置：{device}")
    suffix = re.sub(r"[^\w-]", "_", a.name.strip())
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{suffix}" if suffix else "")
    run_dir = os.path.join(a.out, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    progress = Progress(os.path.join(run_dir, "progress.json"))
    progress.update(force=True, stage="loading", epochs=a.epochs)
    X_train, t_train = load_split(os.path.join(a.data, "train"), progress)
    X_val, t_val = load_split(os.path.join(a.data, "val"), progress)
    targets = [torch.tensor(encode(t)) for t in t_train]

    model = CRNN().to(device)
    progress.say(f"模型參數量：{sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_epoch = (len(X_train) + a.batch - 1) // a.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=steps_per_epoch
    )
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    model_path = os.path.join(run_dir, "plate_crnn.pt")
    history_path = os.path.join(run_dir, "history.json")
    config_path = os.path.join(run_dir, "config.json")
    config = {
        "name": run_name, "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
        "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
        "train_samples": len(X_train), "val_samples": len(X_val),
        "best_epoch": None, "best_val_seq_acc": None, "best_val_char_acc": None,
    }
    progress.say(f"本次訓練存於：{run_dir}")
    history, best = [], -1.0
    train_started = time.time()

    for epoch in range(1, a.epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        perm = torch.randperm(len(X_train))
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            x = train_augment(X_train[idx].to(device))
            tgt = [targets[i] for i in idx.tolist()]
            logits = model(x)
            log_probs = logits.log_softmax(2)
            loss = ctc(
                log_probs,
                torch.cat(tgt),
                torch.full((len(idx),), logits.size(0), dtype=torch.long),
                torch.tensor([len(t) for t in tgt], dtype=torch.long),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(idx)

            batch_no = s // a.batch + 1
            done = (epoch - 1) + batch_no / steps_per_epoch  # 已完成幾個 epoch（含小數）
            elapsed = time.time() - train_started
            progress.update(force=batch_no == steps_per_epoch, stage="training", epoch=epoch,
                            batch=batch_no, batches=steps_per_epoch, elapsed=elapsed,
                            eta=elapsed / done * (a.epochs - done))

        train_loss = total_loss / len(X_train)
        progress.update(force=True, stage="evaluating")
        seq_acc, char_acc = evaluate(model, X_val, t_val, device)
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_seq_acc": round(seq_acc, 4), "val_char_acc": round(char_acc, 4),
        })
        mark = ""
        if seq_acc > best:
            best = seq_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_seq_acc": seq_acc}, model_path)
            config.update(best_epoch=epoch, best_val_seq_acc=round(seq_acc, 4), best_val_char_acc=round(char_acc, 4))
            mark = "  ★ 儲存最佳模型"
            if epoch == 1:  # 第一次存檔後，讓網站切換到這次的模型
                with open(os.path.join(a.out, "active.txt"), "w", encoding="utf-8") as f:
                    f.write(run_name)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=1, ensure_ascii=False)
        progress.say(f"Epoch {epoch:3d}/{a.epochs}  loss {train_loss:.4f}  "
              f"整張正確率 {seq_acc:.2%}  字元正確率 {char_acc:.2%}  "
              f"({time.time() - start:.0f}s){mark}")

    config["finished"] = datetime.now().isoformat(timespec="seconds")
    progress.update(force=True, stage="done", eta=0)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1, ensure_ascii=False)
    progress.say(f"訓練完成，最佳驗證整張正確率 {best:.2%}，模型存於 {model_path}")


if __name__ == "__main__":
    main()
