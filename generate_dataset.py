"""產生合成車牌資料集（仿台灣車牌格式）。

用法：
    python generate_dataset.py                 # 預設 train 30000 張、val 3000 張
    python generate_dataset.py --train 5000 --val 500 --out dataset

輸出：
    dataset/train/*.jpg + dataset/train/labels.csv
    dataset/val/*.jpg   + dataset/val/labels.csv
"""
import argparse
import csv
import os
import random
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from charset import CHARS

LETTERS = [c for c in CHARS if c.isalpha()]  # 台灣車牌不使用 I、O
DIGITS = [c for c in CHARS if c.isdigit()]

# 車牌格式（L=英文、D=數字）與出現權重
FORMATS = [
    ("LLL-DDDD", 0.45),  # 新式自用小客車
    ("LL-DDDD", 0.15),   # 舊式
    ("DDDD-LL", 0.15),   # 舊式
    ("LLL-DDD", 0.15),   # 機車
    ("DDD-LLL", 0.10),
]

# (底色, 字色) 組合
COLOR_SCHEMES = [
    ((245, 245, 240), (20, 20, 20)),    # 白底黑字：自用
    ((245, 245, 240), (190, 30, 30)),   # 白底紅字：營業
    ((245, 245, 240), (30, 120, 60)),   # 白底綠字：電動車
    ((30, 110, 60), (245, 245, 240)),   # 綠底白字：營業大客車
    ((235, 200, 40), (20, 20, 20)),     # 黃底黑字
    ((200, 30, 30), (245, 245, 240)),   # 紅底白字
]
SCHEME_WEIGHTS = [0.55, 0.15, 0.1, 0.08, 0.07, 0.05]

FONT_FILES = [
    "ARIALNB.TTF", "arialbd.ttf", "bahnschrift.ttf", "impact.ttf",
    "consolab.ttf", "courbd.ttf", "calibrib.ttf",
]
FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]

_font_cache = {}


def available_fonts():
    found = []
    for name in FONT_FILES:
        for d in FONT_DIRS:
            path = os.path.join(d, name)
            if os.path.exists(path):
                found.append(path)
                break
    if not found:
        raise RuntimeError("找不到可用字型，請修改 FONT_FILES / FONT_DIRS")
    return found


def get_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def random_text():
    fmt = random.choices([f for f, _ in FORMATS], weights=[w for _, w in FORMATS])[0]
    return "".join(
        random.choice(LETTERS) if c == "L" else random.choice(DIGITS) if c == "D" else c
        for c in fmt
    )


def render_plate(text, fonts):
    """畫出正面、乾淨的車牌。"""
    W, H = 380, 160
    bg, fg = random.choices(COLOR_SCHEMES, weights=SCHEME_WEIGHTS)[0]
    plate = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(plate)

    # 外框
    border = random.randint(3, 7)
    draw.rounded_rectangle([3, 3, W - 4, H - 4], radius=14, outline=fg, width=border) \
        if hasattr(draw, "rounded_rectangle") else \
        draw.rectangle([3, 3, W - 4, H - 4], outline=fg, width=border)

    # 螺絲孔
    if random.random() < 0.5:
        for x in (W * 0.22, W * 0.78):
            draw.ellipse([x - 5, 12, x + 5, 22], fill=(150, 150, 150))

    # 文字：選字型並縮放到適合大小
    font_path = random.choice(fonts)
    size = int(H * random.uniform(0.62, 0.78))
    font = get_font(font_path, size)
    tw, th = draw.textsize(text, font=font)
    max_w = W * random.uniform(0.82, 0.92)
    if tw > max_w:
        size = int(size * max_w / tw)
        font = get_font(font_path, size)
        tw, th = draw.textsize(text, font=font)
    ox, oy = font.getoffset(text)
    x = (W - tw) / 2 + random.uniform(-6, 6)
    y = (H - th - oy) / 2 + random.uniform(4, 12)  # 偏下方，上方留給螺絲孔
    draw.text((x, y - oy / 2), text, font=font, fill=fg)
    return plate


def perspective_coeffs(src, dst):
    """回傳 PIL PERSPECTIVE 轉換係數（把 dst 座標映射回 src）。"""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    return np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)).tolist()


def random_background(w, h):
    base = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    grad = np.linspace(random.uniform(-60, 0), random.uniform(0, 60), w, dtype=np.float32)
    img = base[None, None, :] + grad[None, :, None] + np.random.normal(0, 20, (h, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def augment(plate):
    """透視變形、背景、模糊、光線、雜訊、低解析度、JPEG 壓縮。"""
    W, H = plate.size
    pad = int(W * 0.35)
    cw, ch = W + 2 * pad, H + 2 * pad
    canvas = random_background(cw, ch)

    # 隨機抖動四個角，做透視變形
    j = 0.12
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [
        (pad + x + random.uniform(-j, j) * W, pad + y + random.uniform(-j, j) * H)
        for x, y in src
    ]
    coeffs = perspective_coeffs(src, dst)
    warped = plate.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    mask = Image.new("L", (W, H), 255).transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    canvas.paste(warped, (0, 0), mask)

    # 依車牌四角裁切，四周留隨機邊界
    xs, ys = [p[0] for p in dst], [p[1] for p in dst]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    img = canvas.crop((
        int(min(xs) - random.uniform(0, 0.08) * bw),
        int(min(ys) - random.uniform(0, 0.15) * bh),
        int(max(xs) + random.uniform(0, 0.08) * bw),
        int(max(ys) + random.uniform(0, 0.15) * bh),
    ))

    # 陰影
    if random.random() < 0.3:
        shade = Image.new("L", img.size, 0)
        sw, sh = img.size
        x0 = random.uniform(0, sw)
        ImageDraw.Draw(shade).polygon(
            [(x0, 0), (sw, 0), (sw, sh), (x0 + random.uniform(-sw / 2, sw / 2), sh)],
            fill=random.randint(60, 140),
        )
        img = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, shade)

    # 光線、對比
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.4))

    # 模糊（一般 / 動態）
    r = random.random()
    if r < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
    elif r < 0.45:
        k = [0] * 10 + [1] * 5 + [0] * 10  # 5x5 水平動態模糊
        img = img.filter(ImageFilter.Kernel((5, 5), k, scale=5))

    # 低解析度（模擬遠距離拍攝）
    target_w = random.randint(70, 260)
    target_h = max(16, int(img.height * target_w / img.width))
    img = img.resize((target_w, target_h), Image.BILINEAR)

    # 雜訊
    if random.random() < 0.5:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.normal(0, random.uniform(3, 15), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def make_sample(fonts=None):
    fonts = fonts or available_fonts()
    text = random_text()
    return augment(render_plate(text, fonts)), text


def _worker(args):
    out_dir, start, count, seed = args
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    fonts = available_fonts()
    rows = []
    for i in range(start, start + count):
        img, text = make_sample(fonts)
        name = f"{i:06d}.jpg"
        img.save(os.path.join(out_dir, name), quality=random.randint(35, 95))
        rows.append((name, text))
    return rows


def generate(out_dir, n, workers, seed, force=False):
    if os.path.exists(os.path.join(out_dir, "labels.csv")) and not force:
        print(f"  {out_dir} 已存在，略過（要重新產生請加 --force）")
        return
    os.makedirs(out_dir, exist_ok=True)
    chunk = max(1, n // (workers * 4))
    jobs = [(out_dir, s, min(chunk, n - s), seed + s) for s in range(0, n, chunk)]
    rows = []
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap(_worker, jobs), 1):
            rows.extend(part)
            print(f"\r  {out_dir}: {len(rows)}/{n}", end="", flush=True)
    print()
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="產生合成車牌資料集")
    p.add_argument("--out", default="dataset")
    p.add_argument("--train", type=int, default=30000)
    p.add_argument("--val", type=int, default=3000)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="資料集已存在時仍重新產生並覆蓋")
    a = p.parse_args()
    generate(os.path.join(a.out, "train"), a.train, a.workers, a.seed, a.force)
    generate(os.path.join(a.out, "val"), a.val, a.workers, a.seed + 10_000_000, a.force)
    print("完成！")
