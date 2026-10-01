"""車牌文字辨識模型：CRNN（CNN + 雙向 LSTM）+ CTC。

CNN 抽取影像特徵 → 沿寬度切成 32 個時間步 → BiLSTM 讀序列 → 每步輸出字元機率
CTC 讓模型不需要知道每個字在圖中的確切位置就能訓練。
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from charset import CHARS  # index 0 保留給 CTC blank

IMG_H, IMG_W = 32, 128


def encode(text):
    return [CHARS.index(c) + 1 for c in text]


def preprocess(img):
    """PIL 影像 → (1, 32, 128) 的 tensor。"""
    img = img.convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy((arr - 0.5) / 0.5).unsqueeze(0)


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS) + 1):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(1, 64), nn.MaxPool2d(2, 2),            # 16 x 64
            conv_block(64, 128), nn.MaxPool2d(2, 2),          # 8 x 32
            conv_block(128, 256), conv_block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),                     # 4 x 32
            conv_block(256, 256), nn.MaxPool2d((4, 1), (4, 1)),  # 1 x 32
            nn.Dropout2d(0.2),
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2)        # (N, 256, T)
        f = f.permute(2, 0, 1)            # (T, N, 256)
        out, _ = self.rnn(f)
        return self.fc(out)               # (T, N, C)


def greedy_decode(logits):
    """logits: (T, N, C) → [(text, confidence), ...]"""
    probs = logits.softmax(2).permute(1, 0, 2)  # (N, T, C)
    best_p, best_i = probs.max(2)
    results = []
    for p_seq, i_seq in zip(best_p.tolist(), best_i.tolist()):
        text, conf, prev = [], 1.0, 0
        for p, i in zip(p_seq, i_seq):
            if i != 0 and i != prev:
                text.append(CHARS[i - 1])
                conf *= p
            prev = i
        results.append(("".join(text), conf if text else 0.0))
    return results


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = CRNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
