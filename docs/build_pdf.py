"""產生《車牌辨識系統 教學手冊》PDF。程式碼片段直接從專案原始檔擷取，所以改完程式重跑一次就會更新。

需要額外安裝：pip install "reportlab<4.1"   （matplotlib、pygments 在 Anaconda 已內建）

用法（在專案資料夾執行）：
    python docs/build_pdf.py                         # 訓練曲線用網站目前使用的模型
    python docs/build_pdf.py --run 20261002-000605   # 指定某次訓練

輸出：docs/車牌辨識專案教學.pdf
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime
from xml.sax.saxutils import escape

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "docs", "車牌辨識專案教學.pdf")
TMP = tempfile.mkdtemp()  # 訓練曲線圖片的暫存位置

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String as DString
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

# ---------- 字型 ----------
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("JH", F + "msjh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("JHB", F + "msjhbd.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Mono", F + "consola.ttf"))
pdfmetrics.registerFont(TTFont("MonoB", F + "consolab.ttf"))
pdfmetrics.registerFont(TTFont("Sym", F + "seguisym.ttf"))
pdfmetrics.registerFontFamily("JH", normal="JH", bold="JHB", italic="JH", boldItalic="JHB")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="MonoB", italic="Mono", boldItalic="MonoB")
CMAP = {n: set(pdfmetrics.getFont(n).face.charToGlyph) for n in ("JH", "Mono", "Sym")}


def font_for(ch, base):
    for n in (base, "JH", "Sym"):
        if ord(ch) in CMAP[n]:
            return n
    return "JH"


def mixed(text, base):
    """把 base 字型沒有的字（中文、符號）包上能顯示的字型。text 需是未跳脫的原文。"""
    out, run, run_font = [], "", base
    for ch in text:
        f = base if ord(ch) < 128 else font_for(ch, base)
        if f != run_font and run:
            out.append(escape(run) if run_font == base else f'<font name="{run_font}">{escape(run)}</font>')
            run = ""
        run_font = f
        run += ch
    if run:
        out.append(escape(run) if run_font == base else f'<font name="{run_font}">{escape(run)}</font>')
    return "".join(out)


# ---------- 顏色與樣式 ----------
INK, INK2, MUTED = colors.HexColor("#1f2328"), colors.HexColor("#52514e"), colors.HexColor("#8a8f98")
ACCENT, ACCENT_BG = colors.HexColor("#2a78d6"), colors.HexColor("#eef4fc")
LINE, CODE_BG = colors.HexColor("#d9dce1"), colors.HexColor("#f6f8fa")
TIP_BG, WARN_BG = colors.HexColor("#eef7f0"), colors.HexColor("#fdf4e7")

base = dict(fontName="JH", wordWrap="CJK", textColor=INK)
S = {
    "body": ParagraphStyle("body", fontSize=10.5, leading=18, spaceAfter=7, **base),
    "small": ParagraphStyle("small", fontSize=9, leading=14, textColor=INK2, fontName="JH", wordWrap="CJK"),
    "h1": ParagraphStyle("h1", fontName="JHB", fontSize=20, leading=28, textColor=INK, spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="JHB", fontSize=13.5, leading=20, textColor=INK, spaceBefore=12, spaceAfter=6,
                         keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontSize=10.5, leading=17, leftIndent=14, bulletIndent=3, spaceAfter=3, **base),
    "cell": ParagraphStyle("cell", fontSize=9.2, leading=13.5, **base),
    "cellh": ParagraphStyle("cellh", fontName="JHB", fontSize=9.2, leading=13.5, textColor=INK, wordWrap="CJK"),
    "caption": ParagraphStyle("caption", fontSize=8.8, leading=13, textColor=INK2, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=10, fontName="JH", wordWrap="CJK"),
    "box": ParagraphStyle("box", fontSize=10, leading=16, **base),
}
PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
TEXT_W = PAGE_W - 2 * MARGIN


def code(s):
    """行內程式碼。"""
    return f'<font name="Mono" color="#b4235a">{mixed(s, "Mono")}</font>'


def fallback(markup):
    """在標籤以外的文字中，把正黑體沒有的字（∅、− 等）包上 Sym 字型。"""
    out = []
    for part in re.split(r"(<[^>]+>|&\w+;)", markup):
        if part.startswith("<") or part.startswith("&"):
            out.append(part)
        else:
            out.append("".join(c if ord(c) < 128 or ord(c) in CMAP["JH"] else f'<font name="Sym">{c}</font>'
                               for c in part))
    return "".join(out)


def P(text, style="body"):
    return Paragraph(fallback(text), S[style] if isinstance(style, str) else style)


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def box(text, kind="tip"):
    bg, bar, label = {"tip": (TIP_BG, "#1baf7a", "小提醒"), "warn": (WARN_BG, "#eda100", "注意"),
                      "idea": (ACCENT_BG, "#2a78d6", "想一想")}[kind]
    t = Table([[P(f"<b>{label}</b>　{text}", "box")]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(bar)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [t, Spacer(1, 8)]


def table(rows, widths, header=True):
    data = [[P(c, "cellh" if header and i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * TEXT_W for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0f3")))
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 10)]


# ---------- 程式碼區塊（語法上色） ----------
TOKEN_COLORS = [
    (Comment, "#6a737d"), (String, "#22863a"), (Number, "#b45309"), (Keyword, "#a626a4"),
    (Name.Function, "#2a63c4"), (Name.Class, "#2a63c4"), (Name.Decorator, "#2a63c4"),
    (Name.Builtin, "#b45309"), (Operator.Word, "#a626a4"),
]


def token_color(tt):
    for t, c in TOKEN_COLORS:
        if tt in t:
            return c
    return None


def display_width(line):
    return sum(1.0 if ord(ch) > 0x2000 else 0.55 for ch in line)


def code_block(src, title=None):
    src = src.rstrip("\n")
    markup = []
    for tt, val in lex(src, PythonLexer()):
        m = mixed(val, "Mono")
        c = token_color(tt)
        markup.append(f'<font color="{c}">{m}</font>' if c and val.strip() else m)
    text = "".join(markup).rstrip("\n")
    widest = max(display_width(l) for l in src.split("\n"))
    size = min(8.4, (TEXT_W - 18) / widest)
    style = ParagraphStyle("code", fontName="Mono", fontSize=size, leading=size * 1.42, textColor=INK,
                           backColor=CODE_BG, borderColor=LINE, borderWidth=0.5, borderPadding=(6, 7, 6, 7),
                           spaceBefore=4, spaceAfter=12, leftIndent=7, rightIndent=7)
    out = []
    if title:
        out.append(Paragraph(f'<font name="Mono" color="#52514e">{mixed(title, "Mono")}</font>',
                             ParagraphStyle("ct", fontName="Mono", fontSize=8.3, leading=12, spaceBefore=4,
                                            spaceAfter=7, textColor=INK2, keepWithNext=1)))
    out.append(XPreformatted(text, style))
    return out


def read(path):
    with open(os.path.join(PROJECT, path), encoding="utf-8") as f:
        return f.read()


def snippet(path, start, end=None, include_end=True):
    """從 start 這行（正則）擷取到 end 這行；end=None 時取到下一個頂層定義前。"""
    lines = read(path).split("\n")
    i = next(k for k, l in enumerate(lines) if re.search(start, l))
    if end is None:
        j = i + 1
        while j < len(lines) and not re.match(r"(def |class |@|[A-Z_]+ = |if __name__)", lines[j]):
            j += 1
        while lines[j - 1].strip() == "":
            j -= 1
        return "\n".join(lines[i:j])
    j = next(k for k in range(i + 1, len(lines)) if re.search(end, lines[k]))
    return "\n".join(lines[i:j + (1 if include_end else 0)])


def nlines(path):
    return str(len(read(path).rstrip("\n").split("\n")))


def dedent(s):
    import textwrap
    return textwrap.dedent(s)


# ---------- 圖：流程圖 ----------
def label(d, x, y, text, size=9, font="JH", color=INK, anchor="middle"):
    d.add(DString(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def arrow(d, x1, y1, x2, y2, color=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    L, w = 6, 3.2
    p = [x2, y2,
         x2 - L * math.cos(a) + w * math.sin(a), y2 - L * math.sin(a) - w * math.cos(a),
         x2 - L * math.cos(a) - w * math.sin(a), y2 - L * math.sin(a) + w * math.cos(a)]
    d.add(Polygon(p, fillColor=color, strokeColor=color, strokeWidth=0.5))


def node(d, x, y, w, h, title, sub, kind="code"):
    fill, stroke = (ACCENT_BG, ACCENT) if kind == "code" else (colors.HexColor("#f3f4f6"), colors.HexColor("#9ca3af"))
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    label(d, x + w / 2, y + h - 15, title, 8.4, "Mono" if kind == "code" else "JHB")
    for k, s in enumerate(sub):
        label(d, x + w / 2, y + h - 28 - 11 * k, s, 7.6, "JH", INK2)


def pipeline_diagram():
    W, H = TEXT_W, 170
    d = Drawing(W, H)
    bw, bh = 104, 52
    gap = (W - 4 * bw) / 3
    xs = [k * (bw + gap) for k in range(4)]
    top, bot = 104, 12
    node(d, xs[0], top, bw, bh, "generate_dataset.py", ["畫出合成車牌", "+ 資料增強"])
    node(d, xs[1], top, bw, bh, "dataset/", ["train 30000 張", "val 3000 張 + labels.csv"], "data")
    node(d, xs[2], top, bw, bh, "train.py", ["CRNN + CTC 訓練", "每個 epoch 驗證"])
    node(d, xs[3], top, bw, bh, "models/runs/", ["每次訓練一個資料夾", "plate_crnn.pt 等"], "data")
    for k in range(3):
        arrow(d, xs[k] + bw + 1, top + bh / 2, xs[k + 1] - 1, top + bh / 2)
    node(d, xs[2], bot, bw, bh, "app.py", ["Flask 網站後端", "載入模型、提供 API"])
    node(d, xs[3], bot, bw, bh, "瀏覽器", ["index.html + app.js", "用 HTTP / JSON 溝通"], "data")
    node(d, xs[0], bot, bw * 2 + gap, bh, "plate_model.py / charset.py",
         ["共用：模型結構、影像前處理、字元集、解碼", "train.py 與 app.py 都 import 它們"])
    arrow(d, xs[3] + bw / 2, top - 1, xs[2] + bw * 0.78, bot + bh + 1)
    label(d, xs[3] + bw / 2 + 2, (top + bot + bh) / 2 - 3, "讀取模型", 7.6, "JH", INK2, "start")
    arrow(d, xs[2] + bw + 1, bot + bh / 2 + 6, xs[3] - 1, bot + bh / 2 + 6)
    arrow(d, xs[3] - 1, bot + bh / 2 - 6, xs[2] + bw + 1, bot + bh / 2 - 6)
    arrow(d, xs[2] + 20, bot + bh + 1, xs[2] + 20, top - 1)
    label(d, xs[2] + 16, (top + bot + bh) / 2 - 3, "網頁按「訓練」時啟動", 7.6, "JH", INK2, "end")
    return d


def crnn_diagram():
    rows = [
        ("輸入影像（灰階）", "1 × 32 × 128", "preprocess()：縮放成 32×128、正規化到 -1~1"),
        ("conv 64 → MaxPool 2×2", "64 × 16 × 64", "高、寬都減半"),
        ("conv 128 → MaxPool 2×2", "128 × 8 × 32", "高、寬再減半；寬度 32 從這裡固定下來"),
        ("conv 256 ×2 → MaxPool (2,1)", "256 × 4 × 32", "只壓高度，不壓寬度"),
        ("conv 256 → MaxPool (4,1)", "256 × 1 × 32", "高度壓成 1"),
        ("squeeze + permute", "32 × N × 256", "寬度方向的 32 格 = 32 個時間步"),
        ("雙向 LSTM ×2 層（hidden 128）", "32 × N × 256", "左→右、右→左各 128 維，接起來 256"),
        ("Linear 256 → 36", "32 × N × 36", "每個時間步：35 個字元 + 1 個 blank 的分數"),
    ]
    W, rh = TEXT_W, 25
    H = rh * len(rows) + 6
    d = Drawing(W, H)
    bw = 172
    for k, (name, shape, note) in enumerate(rows):
        y = H - (k + 1) * rh
        cnn = 1 <= k <= 4
        fill = ACCENT_BG if cnn else (colors.HexColor("#fdf0ea") if k >= 6 else colors.HexColor("#f3f4f6"))
        stroke = ACCENT if cnn else (colors.HexColor("#eb6834") if k >= 6 else colors.HexColor("#9ca3af"))
        d.add(Rect(0, y + 3, bw, rh - 8, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
        label(d, bw / 2, y + 9.5, name, 8.3, "JH")
        label(d, bw + 12, y + 9.5, shape, 8.6, "MonoB", INK, "start")
        label(d, bw + 100, y + 9.5, note, 8, "JH", INK2, "start")
        if k:
            arrow(d, bw / 2, y + rh + 2.5, bw / 2, y + rh - 4.5)
    return d


def ctc_diagram():
    frames = list("AA_B__-11_12")
    W, H = TEXT_W, 128
    d = Drawing(W, H)
    cw, x0 = 26, 92
    def row(y, title, cells, hl=None, note=None):
        label(d, 0, y + 7, title, 8.6, "JHB", INK, "start")
        for k, c in enumerate(cells):
            if c is None:
                continue
            x = x0 + k * cw
            on = hl is None or hl[k]
            d.add(Rect(x, y, cw - 4, 22, rx=3, ry=3,
                       fillColor=(ACCENT_BG if on else colors.HexColor("#f3f4f6")),
                       strokeColor=(ACCENT if on else colors.HexColor("#c4c8ce")), strokeWidth=0.7))
            label(d, x + (cw - 4) / 2, y + 7, "∅" if c == "_" else c, 10, "MonoB" if c != "_" else "Sym",
                  INK if on else MUTED)
        if note:
            label(d, x0 + len(cells) * cw + 4, y + 7, note, 8, "JH", INK2, "start")
    row(96, "① 每步取最大", frames, note="T = 12（實際 32）")
    keep, prev = [], None
    for c in frames:
        keep.append(c != prev)
        prev = c
    row(56, "② 合併連續重複", frames, keep, note="灰色 = 重複，丟掉")
    out = [c for c, k in zip(frames, keep) if k and c != "_"]
    row(16, "③ 去掉 blank", out, note="→ 結果「AB-112」")
    return d


# ---------- 圖：訓練曲線（真實資料） ----------
def pick_run(run):
    """沒有指定時，用網站目前使用的模型（models/active.txt）。"""
    runs_dir = os.path.join(PROJECT, "models", "runs")
    if not run:
        active = os.path.join(PROJECT, "models", "active.txt")
        if not os.path.exists(active):
            sys.exit("找不到 models/active.txt，請先訓練模型，或用 --run 指定")
        run = open(active, encoding="utf-8").read().strip()
    if not os.path.exists(os.path.join(runs_dir, run, "history.json")):
        sys.exit(f"找不到 models/runs/{run}/history.json")
    return run


def charts(run):
    hist = json.load(open(os.path.join(PROJECT, "models", "runs", run, "history.json"), encoding="utf-8"))
    ep = [h["epoch"] for h in hist]
    fp = font_manager.FontProperties(fname=F + "msjh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    def style(ax, title):
        ax.set_title(title, fontproperties=fp, fontsize=10.5, loc="left", color="#0b0b0b", pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9ccd1")
        ax.tick_params(colors="#52514e", labelsize=8, length=0)
        ax.grid(axis="y", color="#e6e8eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(ep)
        ax.set_xlabel("epoch", fontsize=8, color="#52514e")

    fig, ax = plt.subplots(figsize=(3.3, 2.35), dpi=220)
    ax.plot(ep, [h["train_loss"] for h in hist], color="#2a78d6", lw=2, marker="o", ms=4.5,
            markeredgecolor="white", markeredgewidth=1)
    style(ax, "訓練 loss（越低越好）")
    for h in (hist[0], hist[-1]):
        ax.annotate(f'{h["train_loss"]:.2f}', (h["epoch"], h["train_loss"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#0b0b0b")
    ax.set_ylim(0, max(h["train_loss"] for h in hist) * 1.18)
    fig.tight_layout()
    paths.append(os.path.join(TMP, "chart_loss.png"))
    fig.savefig(paths[-1])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 2.35), dpi=220)
    series = [("val_char_acc", "字元正確率", "#eb6834"), ("val_seq_acc", "整張正確率", "#2a78d6")]
    for key, name, c in series:
        ys = [h[key] * 100 for h in hist]
        ax.plot(ep, ys, color=c, lw=2, marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=1, label=name)
        ax.annotate(f"{ys[-1]:.1f}%", (ep[-1], ys[-1]), textcoords="offset points", xytext=(0, 7 if key == "val_char_acc" else -13),
                    ha="center", fontsize=8, color="#0b0b0b")
    style(ax, "驗證正確率")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    leg = ax.legend(prop=font_manager.FontProperties(fname=F + "msjh.ttc", size=7.5), frameon=False,
                    loc="upper left", handlelength=1.6)
    for t in leg.get_texts():
        t.set_color("#52514e")
    fig.tight_layout()
    paths.append(os.path.join(TMP, "chart_acc.png"))
    fig.savefig(paths[-1])
    plt.close(fig)
    return hist, paths


def sample_grid(n=8, cols=4):
    with open(os.path.join(PROJECT, "dataset", "val", "labels.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n]
    cw = TEXT_W / cols
    cells = []
    for r in rows:
        path = os.path.join(PROJECT, "dataset", "val", r["filename"])
        from PIL import Image as PImage
        w, h = PImage.open(path).size
        iw = cw - 14
        ih = min(18 * mm, iw * h / w)
        iw = ih * w / h
        cells.append([Image(path, width=iw, height=ih),
                      P(f'<font name="MonoB">{r["text"]}</font>　<font color="#8a8f98">{r["filename"]}</font>', "caption")])
    grid = [cells[k:k + cols] for k in range(0, len(cells), cols)]
    data = []
    for g in grid:
        data.append([c[0] for c in g])
        data.append([c[1] for c in g])
    t = Table(data, colWidths=[cw] * cols)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


# ---------- 文件 ----------
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=22 * mm,
                         bottomMargin=20 * mm, title="車牌辨識系統 教學手冊", author="Peter-Project",
                         subject="CRNN + CTC 車牌文字辨識：程式碼導讀")
        frame = Frame(MARGIN, 20 * mm, TEXT_W, PAGE_H - 42 * mm, id="f", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("body", [frame], onPage=self.decorate)])

    def decorate(self, c, doc):
        c.saveState()
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(MARGIN, PAGE_H - 14 * mm, PAGE_W - MARGIN, PAGE_H - 14 * mm)
        c.setFont("JH", 8)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, PAGE_H - 12 * mm, "車牌辨識系統 教學手冊")
        c.drawRightString(PAGE_W - MARGIN, 12 * mm, str(doc.page))
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            level = 0 if f.style.name == "h1" else 1
            text = re.sub(r"<[^>]+>", "", f.getPlainText())
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def H1(t):
    return [CondPageBreak(60 * mm), P(t, "h1"),
            Table([[""]], colWidths=[TEXT_W], rowHeights=[2],
                  style=[("LINEABOVE", (0, 0), (-1, 0), 1.5, ACCENT)]), Spacer(1, 8)]


def H2(t):
    return [CondPageBreak(35 * mm), P(t, "h2")]


def build(run):
    hist, chart_paths = charts(run)
    cfg = json.load(open(os.path.join(PROJECT, "models", "runs", run, "config.json"), encoding="utf-8"))
    best = max(hist, key=lambda h: h["val_seq_acc"])
    per_epoch = ""
    if cfg.get("finished"):
        sec = (datetime.fromisoformat(cfg["finished"]) - datetime.fromisoformat(cfg["started"])).total_seconds()
        per_epoch = f"，每個 epoch 約 {sec / len(hist):.0f} 秒"
    story = []

    # ===== 封面 =====
    from reportlab.platypus import NextPageTemplate
    story += [Spacer(1, 38 * mm),
              P('<font color="#2a78d6">PETER-PROJECT</font>', ParagraphStyle("k", fontName="MonoB", fontSize=11, leading=14)),
              Spacer(1, 6),
              P("車牌辨識系統<br/>教學手冊", ParagraphStyle("t", fontName="JHB", fontSize=34, leading=46, textColor=INK)),
              Spacer(1, 8),
              P("從合成資料、CRNN + CTC 模型、訓練流程，到 Flask 網站的程式碼導讀",
                ParagraphStyle("st", fontName="JH", fontSize=13, leading=20, textColor=INK2, wordWrap="CJK")),
              Spacer(1, 14 * mm), sample_grid(8, 4), Spacer(1, 4),
              P("以上是專案自己產生的驗證集圖片（dataset/val），底下是正確答案與檔名。", "caption"),
              Spacer(1, 16 * mm),
              P(f"文件產生日期：{date.today().isoformat()}", "small"),
              NextPageTemplate("body"), PageBreak()]

    # ===== 目錄 =====
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="JHB", fontSize=11, leading=18, leftIndent=0, textColor=INK),
        ParagraphStyle("toc1", fontName="JH", fontSize=9.5, leading=14, leftIndent=16, textColor=INK2),
    ]
    toc.dotsMinLevel = 0
    story += [P("目錄", S["h1"].clone("toc_title")), Spacer(1, 6), toc, PageBreak()]

    # ===== 1. 專案簡介 =====
    story += H1("1. 這個專案在做什麼")
    story += [P("這個專案用 PyTorch <b>從零開始</b>訓練一個「車牌文字辨識」模型：給它一張已經裁切好的車牌照片，"
                "它會讀出上面的文字，例如 " + code("ABC-1234") + "。整個流程不需要下載任何資料集——"
                "訓練資料是程式自己畫出來的合成車牌；訓練好的模型則透過一個 Flask 網站展示，"
                "可以上傳照片、隨機測試、看訓練曲線，甚至直接在網頁上按按鈕訓練。")]
    story += [Spacer(1, 4), pipeline_diagram(),
              P("圖 1　整體架構：藍框是程式，灰框是資料或畫面。", "caption")]
    story += [P("讀完這份手冊，你會了解：")]
    story += bullets(["怎麼用程式產生大量、帶有正確答案的訓練圖片，以及資料增強為什麼重要",
                      "CRNN（CNN + 雙向 LSTM）怎麼把一張圖變成一串字元的機率",
                      "CTC Loss 如何讓模型在<b>不知道每個字確切位置</b>的情況下學會讀字",
                      "一個完整的 PyTorch 訓練迴圈：資料、loss、optimizer、learning rate 排程、驗證、存檔",
                      "怎麼把模型包成網站，並讓網頁能啟動訓練、即時顯示進度"])
    story += H2("檔案一覽")
    story += table([
        ["檔案", "行數", "負責什麼"],
        [code("charset.py"), nlines("charset.py"), "字元集：0–9、A–Z（不含 I、O）和 " + code("-") + "，共 35 個字元"],
        [code("generate_dataset.py"), nlines("generate_dataset.py"), "畫出合成車牌、做資料增強，輸出 " + code("dataset/") + " 的圖片與 " + code("labels.csv")],
        [code("plate_model.py"), nlines("plate_model.py"), "模型結構（CRNN）、影像前處理、文字編碼與解碼、載入模型"],
        [code("train.py"), nlines("train.py"), "訓練流程：讀資料、訓練、驗證、存最佳模型、寫訓練紀錄與進度"],
        [code("app.py"), nlines("app.py"), "Flask 網站後端：辨識 API、模型管理、從網頁啟動產生資料／訓練"],
        [code("templates/index.html"), nlines("templates/index.html"), "網頁的版面"],
        [code("static/app.js"), nlines("static/app.js"), "網頁的互動：框選車牌、呼叫 API、畫訓練曲線與進度條"],
        [code("static/style.css"), nlines("static/style.css"), "網頁樣式"],
    ], [0.27, 0.08, 0.65])

    # ===== 2. 快速開始 =====
    story += H1("2. 快速開始")
    story += [P("所有指令都在專案資料夾 " + code(r"D:\Peter Project") + " 裡執行。")]
    story += code_block(dedent("""\
        pip install -r requirements.txt     # flask、torch、numpy、Pillow

        # 1. 產生資料集（預設 train 30000 張、val 3000 張；已存在會自動略過）
        python generate_dataset.py

        # 2. 訓練模型（每次訓練存成 models/runs/<日期-時間>-<名稱>/）
        python train.py --epochs 30 --name baseline

        # 3. 開網站，然後用瀏覽器打開 http://127.0.0.1:5000
        python app.py"""), "終端機")
    story += box("不能直接雙擊 " + code("templates/index.html") + " 打開：它是 Flask 樣板，裡面的 "
                 + code("{{ url_for(...) }}") + " 要由伺服器換成真正的路徑，而且辨識要靠後端的 PyTorch 執行。"
                 "一定要先跑 " + code("python app.py") + "。", "warn")
    story += [P("步驟 1、2 也可以不打指令：開好網站後，在最上方的「訓練工作台」設定參數按按鈕即可。網站的各區塊：")]
    story += table([
        ["區塊", "可以做什麼"],
        ["訓練工作台", "產生資料集、設定 epochs / batch size / learning rate 後開始訓練；右側有進度條和執行紀錄"],
        ["已儲存的模型", "列出每次訓練的參數和最佳成績，按「使用」切換網站要用的模型"],
        ["上傳圖片辨識", "上傳照片，在圖上拖曳框出車牌再按辨識（模型只會讀字，不會自己找車牌）"],
        ["隨機產生測試車牌", "即時畫一張模型沒看過的車牌來考它"],
        ["訓練曲線", "loss 和驗證正確率隨 epoch 的變化，訓練中會自動更新"],
        ["驗證集抽樣結果", "隨機抽 24 張驗證圖片，用綠框／紅框標出對錯"],
    ], [0.24, 0.76])
    story += [P(f"以模型 {run} 為例：用 {cfg['train_samples']} 張圖訓練 {len(hist)} 個 epoch{per_epoch}，"
                f"最好的整張正確率達到 <b>{best['val_seq_acc']:.1%}</b>、字元正確率 <b>{best['val_char_acc']:.1%}</b>"
                f"（詳見第 6 章的訓練曲線）。")]

    # ===== 3. 字元集與資料 =====
    story += H1("3. 合成資料集：generate_dataset.py")
    story += [P("深度學習需要大量「圖片 + 正確答案」的配對。真實車牌照片既難收集、又得一張張人工標註；"
                "這個專案改用程式<b>直接畫出車牌</b>——因為文字是我們自己決定的，答案天生就是對的，要幾張就有幾張。")]
    story += H2("3.1 字元集")
    story += code_block(read("charset.py"), "charset.py")
    story += [P("台灣車牌不使用 I 和 O（避免和 1、0 混淆），所以字元集是 10 個數字 + 24 個英文字母 + "
                + code("-") + "，共 35 個。模型輸出時會再多一類 <b>blank</b>（第 5 章會解釋），所以總共 36 類。")]
    story += H2("3.2 車牌格式與顏色")
    story += code_block(snippet("generate_dataset.py", r"^FORMATS = ", r"^SCHEME_WEIGHTS"), "generate_dataset.py")
    story += [P(code("random_text()") + " 先依權重抽一種格式，再把 " + code("L") + " 換成隨機英文字母、"
                + code("D") + " 換成隨機數字。顏色組合也一樣依權重抽，所以常見的白底黑字最多。")]
    story += H2("3.3 畫出乾淨的車牌：render_plate()")
    story += code_block(snippet("generate_dataset.py", r"^def render_plate"), "generate_dataset.py")
    story += [P("在 380×160 的畫布上畫底色、圓角外框、（一半機率）螺絲孔，再從 7 種 Windows 內建字型中隨機挑一種寫字。"
                "字太寬時會等比例縮小，位置也加了一點隨機偏移，讓模型不會只認得「永遠在正中央」的字。")]
    story += H2("3.4 資料增強：augment()")
    story += [P("乾淨的車牌和真實照片差很多。" + code("augment()") + " 依序加上各種「真實世界的髒東西」：")]
    story += table([
        ["步驟", "做法", "模擬什麼"],
        ["透視變形", "隨機抖動四個角，用 " + code("Image.PERSPECTIVE") + " 變形", "從斜角拍攝"],
        ["隨機背景", "隨機底色 + 漸層 + 雜訊", "車牌周圍的車身、路面"],
        ["隨機裁切邊界", "四周多留 0–15% 的背景", "人工框選不精準"],
        ["陰影", "30% 機率蓋上半透明多邊形", "樹影、建築物陰影"],
        ["亮度 / 對比", "各乘上 0.5–1.4 倍", "白天、夜晚、逆光"],
        ["模糊", "30% 高斯模糊、15% 水平動態模糊", "對焦失準、車子在移動"],
        ["低解析度", "縮到寬 70–260 像素", "遠距離拍攝"],
        ["雜訊 + JPEG", "50% 加高斯雜訊；存檔品質 35–95", "感光元件雜訊、壓縮失真"],
    ], [0.18, 0.5, 0.32])
    story += [Spacer(1, 2), sample_grid(8, 4),
              P("圖 2　增強後的真實輸出（dataset/val 前 8 張）。有些已經模糊到人眼都要看一下。", "caption")]
    story += code_block(snippet("generate_dataset.py", r"^def augment"), "generate_dataset.py")
    story += box("透視變形用的 " + code("perspective_coeffs()") + " 是在解一個 8 元一次聯立方程式：已知四個角變形前後的座標，"
                 "求出能把新座標映射回原座標的 8 個係數。這正是 PIL " + code("PERSPECTIVE") + " 需要的參數。", "idea")
    story += H2("3.5 多行程平行產生")
    story += code_block(snippet("generate_dataset.py", r"^def generate"), "generate_dataset.py")
    story += [P("把 n 張圖切成很多小塊，用 " + code("multiprocessing.Pool") + " 讓多個 CPU 核心同時畫圖。"
                "每一塊用不同的 " + code("seed") + "（亂數種子），所以結果<b>可重現</b>：同樣的參數再跑一次，會得到一模一樣的資料集。"
                "驗證集的 seed 加了 10,000,000，確保和訓練集不重複。")]

    # ===== 4/5. 模型 =====
    story += H1("4. 模型：plate_model.py")
    story += H2("4.1 影像前處理與文字編碼")
    story += code_block(snippet("plate_model.py", r"^def encode") + "\n\n\n" + snippet("plate_model.py", r"^def preprocess"),
                        "plate_model.py")
    story += [P("不管原圖多大，一律轉成灰階、縮放成 <b>高 32 × 寬 128</b>，再把像素值從 0–255 換成 -1～1。"
                "文字則轉成數字：" + code("encode('AB-1')") + " 會得到每個字元在 " + code("CHARS")
                + " 裡的位置 <b>+1</b>，因為編號 0 要保留給 CTC 的 blank。")]
    story += H2("4.2 CRNN 的結構")
    story += code_block(snippet("plate_model.py", r"^def conv_block") + "\n\n\n" + snippet("plate_model.py", r"^class CRNN"),
                        "plate_model.py")
    story += [P("CRNN = <b>C</b>onvolutional + <b>R</b>ecurrent <b>N</b>eural <b>N</b>etwork。"
                "關鍵是 CNN 的池化設計：前兩次池化把高和寬都減半，之後只壓高度、不壓寬度，"
                "最後得到高度 1、寬度 32 的特徵圖。寬度方向的每一格，對應原圖由左到右的一個窄條，"
                "就像把車牌切成 32 個時間步，交給 LSTM 依序讀。")]
    story += [crnn_diagram(), P("圖 3　一張圖在模型裡的形狀變化（N = batch size）。藍色是 CNN，橘色是序列部分。", "caption")]
    story += [P("為什麼要用<b>雙向</b> LSTM？判斷某一格是什麼字時，看看左右兩邊的內容很有幫助——"
                "例如一個筆畫被陰影遮住一半，鄰近幾格的資訊能幫忙補回來。整個模型約 <b>235 萬</b>個參數"
                "（訓練時會印出 " + code("模型參數量：2,351,524") + "）。")]

    story += H1("5. CTC：不用標位置也能學會讀字")
    story += [P("模型對每張圖輸出 32 個時間步，但車牌只有 6–8 個字，而且我們<b>不知道</b>每個字落在第幾步。"
                "CTC（Connectionist Temporal Classification）的解法是：多加一個特殊類別 <b>blank</b>（記作 ∅，編號 0），"
                "並規定輸出要經過「合併連續重複 → 刪除 blank」才變成最終文字。")]
    story += [ctc_diagram(), P("圖 4　CTC 解碼的例子。注意中間的 ∅ 讓兩個「1」得以保留成「11」。", "caption")]
    story += [P("訓練時，" + code("nn.CTCLoss") + " 會把<b>所有</b>能解碼成正確答案的路徑（例如「AAB」「A∅B」「∅AB」…）"
                "的機率加總，並讓這個總機率越大越好。模型因此自己學會字在哪裡，我們只需要提供整串答案。")]
    story += H2("5.1 推論：greedy_decode()")
    story += code_block(snippet("plate_model.py", r"^def greedy_decode"), "plate_model.py")
    story += [P("每個時間步取機率最高的類別（greedy），然後套用上面的規則：" + code("i != 0") + " 去掉 blank、"
                + code("i != prev") + " 合併重複。信心度是所有輸出字元機率的乘積，網頁上顯示的「信心度」就是它。")]
    story += box("CHARS 裡本來就有 " + code("-") + " 這個字元，它是車牌上的連字號，和 blank 是兩回事。"
                 "這也是 blank 不用「-」表示、而在圖 4 用 ∅ 的原因。", "tip")

    # ===== 6. 訓練 =====
    story += H1("6. 訓練流程：train.py")
    story += H2("6.1 準備")
    story += code_block(dedent(snippet("train.py", r"model = CRNN\(\)\.to\(device\)", r"ctc = nn\.CTCLoss")), "train.py（main 內）")
    story += table([
        ["元件", "設定", "作用"],
        ["Adam", "lr 預設 1e-3", "根據梯度更新參數的演算法，對學習率不太敏感，很適合入門"],
        ["OneCycleLR", "max_lr = --lr", "學習率先由小升到最大、再慢慢降到很小。前期跑得快、後期收得穩"],
        ["CTCLoss", "blank=0, zero_infinity=True", "第 5 章的 CTC；zero_infinity 避免某些不可能的對齊產生無限大 loss"],
    ], [0.17, 0.28, 0.55])
    story += H2("6.2 訓練迴圈")
    story += code_block(dedent(snippet("train.py", r"for s in range\(0, len\(perm\), a\.batch\)", r"total_loss \+= loss")),
                        "train.py（每個 batch）")
    story += [P("這就是 PyTorch 訓練的標準五步：<b>前向</b>（" + code("model(x)") + "）→ <b>算 loss</b> → "
                + code("zero_grad()") + " 清掉舊梯度 → " + code("backward()") + " 反向傳播 → "
                + code("optimizer.step()") + " 更新參數。另外兩個細節：")]
    story += bullets([code("clip_grad_norm_(..., 5)") + "：梯度太大時等比例縮小，防止 LSTM 偶爾的梯度爆炸讓訓練崩掉。",
                      code("CTCLoss") + " 需要四樣東西：log 機率、所有答案接成一長串、每張圖的時間步數（都是 32）、每個答案的長度。"])
    story += [P("每個 batch 送進模型前，還會再做一次<b>即時</b>資料增強：")]
    story += code_block(snippet("train.py", r"^def train_augment"), "train.py")
    story += [P("同一張圖在不同 epoch 會有不同的亮度和位置，等於免費多出很多資料。")]
    story += H2("6.3 驗證：兩種正確率")
    story += code_block(snippet("train.py", r"^def edit_distance") + "\n\n\n" + snippet("train.py", r"^def evaluate"), "train.py")
    story += table([
        ["指標", "算法", "例子：答案 ABC-1234，模型讀成 ABC-1284"],
        ["整張正確率", "完全一樣才算對", "0%（這張算錯）"],
        ["字元正確率", "1 − 編輯距離總和 / 答案總字數", "1 − 1/8 = 87.5%（只錯 1 個字）"],
    ], [0.17, 0.38, 0.45])
    story += [P("編輯距離（Levenshtein distance）是「最少要改幾個字才會一樣」，包含替換、插入、刪除。"
                "實際應用看的是整張正確率——車牌錯一個字就是錯的車；字元正確率則能看出模型「差多少」。")]
    story += H2("6.4 你的真實訓練結果")
    imgs = [Image(p, width=TEXT_W / 2 - 4, height=(TEXT_W / 2 - 4) * 2.35 / 3.3) for p in chart_paths]
    t = Table([imgs], colWidths=[TEXT_W / 2] * 2)
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, P(f"圖 5　models/runs/{run}：{len(hist)} 個 epoch、batch {cfg['batch']}、lr {cfg['lr']}。", "caption")]
    story += table([["Epoch", "訓練 loss", "整張正確率", "字元正確率"]] +
                   [[str(h["epoch"]), f'{h["train_loss"]:.4f}', f'{h["val_seq_acc"]:.2%}', f'{h["val_char_acc"]:.2%}'] for h in hist],
                   [0.16, 0.28, 0.28, 0.28])
    story += [P("前幾個 epoch 整張正確率常常是 0%，這很正常：CTC 模型一開始常只會輸出 blank 或少數幾個字，"
                "要先學會「字大概在哪」，整張正確率才會突然開始上升。只訓練 1 個 epoch 的模型幾乎全錯，原因就在這裡。")]
    story += H2("6.5 存檔與訓練進度")
    story += code_block(dedent("""\
        models/
        |-- active.txt               網站目前使用哪個模型
        `-- runs/
            `-- 20261002-000605/      每次訓練一個資料夾，不會互相覆蓋
                |-- plate_crnn.pt    驗證整張正確率最高那個 epoch 的權重
                |-- history.json     每個 epoch 的 loss / 正確率（網頁畫曲線用）
                |-- config.json      訓練參數與最佳成績（網頁「已儲存的模型」表格用）
                `-- progress.json    即時進度（網頁進度條用，最多每秒更新一次）"""), "資料夾結構")
    story += [P("每個 epoch 驗證完，只有在整張正確率<b>創新高</b>時才覆寫 " + code("plate_crnn.pt")
                + "，所以中途按 Ctrl+C 停掉也不會失去最好的那一版。第一次存檔時還會改寫 " + code("active.txt")
                + "，讓網站自動換成這次的新模型。")]
    story += [P(code("Progress") + " 類別負責回報進度：同時寫 " + code("progress.json")
                + "（給網頁的進度條），並在終端機用 " + code(r"\r") + " 覆寫同一行印出文字進度條：")]
    story += code_block("  Epoch 2/5 [###########---------]  55%  batch 520/938  預估剩餘 01:40", "終端機輸出")
    story += box("寫 " + code("progress.json") + " 時先寫到 " + code(".tmp") + " 再用 " + code("os.replace()")
                 + " 換過去。這樣網站讀檔時，看到的永遠是完整的舊檔或新檔，不會讀到寫一半的內容。", "tip")
    story += box(".gitignore 排除了 " + code("models/") + " 和 " + code("dataset/") + "，訓練好的模型只存在這台電腦，不會進 git。"
                 "要備份或分享模型，請自行複製該次的 run 資料夾。", "warn")

    # ===== 7. 網站 =====
    story += H1("7. 網站：app.py 與前端")
    story += H2("7.1 API 一覽")
    story += table([
        ["路由", "方法", "做什麼"],
        [code("/"), "GET", "回傳網頁（index.html）"],
        [code("/api/status"), "GET", "資料集張數、目前使用的模型與成績"],
        [code("/api/models"), "GET", "列出所有已儲存的模型"],
        [code("/api/models/active"), "POST", "切換網站使用的模型（改寫 active.txt）"],
        [code("/api/recognize"), "POST", "上傳圖片 → 回傳辨識文字與信心度"],
        [code("/api/random"), "GET", "即時產生一張合成車牌並辨識，回傳圖片（base64）、答案、結果"],
        [code("/api/val-samples"), "GET", "從驗證集隨機抽 n 張辨識"],
        [code("/api/history"), "GET", "某次訓練的 history.json"],
        [code("/api/progress"), "GET", "最近一次訓練的 progress.json，加上「幾秒前更新」"],
        [code("/api/jobs/generate"), "POST", "在背景執行 generate_dataset.py"],
        [code("/api/jobs/train"), "POST", "在背景執行 train.py"],
        [code("/api/jobs/stop"), "POST", "停止背景工作"],
        [code("/api/jobs"), "GET", "背景工作狀態與最後 40 行執行紀錄"],
    ], [0.33, 0.1, 0.57])
    story += H2("7.2 模型熱更新")
    story += code_block(snippet("app.py", r"^def get_model") + "\n\n\n" + snippet("app.py", r"^def recognize"), "app.py")
    story += [P("每次辨識前，用「模型名稱 + 檔案修改時間」當作 key。只要切換了模型、或訓練中 "
                + code("plate_crnn.pt") + " 被更新，key 就會變，於是自動重新載入——所以訓練時開著網站，辨識結果會越來越準。"
                + code("lock") + " 確保多個請求同時進來時，不會兩個人同時在載入模型。")]
    story += H2("7.3 從網頁啟動訓練")
    story += code_block(snippet("app.py", r"^def start_job"), "app.py")
    story += [P("網頁按「開始訓練」時，後端用 " + code("subprocess.Popen") + " 另開一個 Python 行程執行 "
                + code("train.py") + "，輸出寫進 " + code("logs/job.log") + "。" + code("-u")
                + " 讓輸出不經緩衝，網頁才能即時看到。同一時間只允許一個工作，避免兩個訓練搶 GPU。")]
    story += [P("前端（" + code("static/app.js") + "）則是用<b>輪詢</b>：每 2 秒問一次 " + code("/api/jobs")
                + " 和 " + code("/api/progress") + " 更新執行紀錄與進度條，每 10 秒更新一次訓練曲線。"
                "超過 60 秒沒有新進度，進度條會變灰並顯示「訓練已停止或中斷」。")]
    story += H2("7.4 上傳與框選")
    story += [P("上傳的照片畫在 " + code("<canvas>") + " 上，拖曳滑鼠時記錄框選範圍（換算回原圖座標）。"
                "按「辨識」時，前端先把框選範圍裁切成新圖，再以 PNG 上傳到 " + code("/api/recognize")
                + "——也就是說，<b>裁切是在瀏覽器裡做的</b>，後端收到的已經是車牌本身。")]

    # ===== 8. 實驗 =====
    story += H1("8. 動手實驗")
    story += [P("理解程式碼最好的方法是改改看。每次訓練都會存成獨立資料夾，可以放心實驗，再到網頁上比較曲線：")]
    story += table([
        ["實驗", "怎麼做", "觀察什麼"],
        ["訓練久一點", code("python train.py --epochs 30 --name e30"), "正確率能到多高？什麼時候開始變平？"],
        ["學習率", code("--lr 3e-3 --name lr3e-3") + " 和 " + code("--lr 3e-4"), "太大會不穩定、太小學得慢"],
        ["資料量", "產生 5000 張 vs 50000 張（加 " + code("--force") + "）", "資料越多越好嗎？差多少？"],
        ["拿掉資料增強", "把 " + code("augment()") + " 改成只做縮放", "驗證集分數變高，但真實照片變差——為什麼？"],
        ["縮小模型", "把 LSTM 改成 1 層、通道數減半", "參數少很多，正確率掉多少？"],
        ["真實照片", "拍幾張真的車牌上傳測試", "找出模型的弱點，回頭調整資料產生方式"],
    ], [0.18, 0.45, 0.37])
    story += H2("目前的限制")
    story += bullets(["只做<b>文字辨識</b>，需要先框選車牌。完整系統還需要一個「車牌偵測」模型（例如 YOLO）先找出車牌位置。",
                      "訓練資料全是合成的，和真實照片仍有差距：反光、髒污、螺絲遮擋、特殊字型、新式車牌的細節等。",
                      "驗證集也是合成的，所以高分只代表「很會讀合成車牌」，真實表現要用真實照片測。"])
    story += H2("常見問題")
    story += table([
        ["狀況", "原因與解法"],
        ["網頁看不到進度條或新功能", "瀏覽器還在用舊版 app.js。按一次 Ctrl + F5 強制重新整理。"],
        ["辨識結果幾乎全錯", "模型訓練太少。只跑 1–2 個 epoch 時整張正確率通常是 0%，至少訓練 5 個以上。"],
        ["網頁打不開", "app.py 沒有在跑。開終端機執行 " + code("python app.py") + "，並保持視窗開著。"],
        ["按「產生資料集」是灰色的", "資料集已經存在。要重產請在終端機加 " + code("--force") + "。"],
        ["進度條變灰、顯示中斷", "超過 1 分鐘沒有新進度：訓練被停止、當掉，或電腦休眠了。看執行紀錄最後幾行找原因。"],
    ], [0.3, 0.7])

    # ===== 名詞表 =====
    story += H1("附錄：名詞對照")
    story += table([
        ["名詞", "說明"],
        ["epoch", "把整個訓練集完整看過一遍"],
        ["batch / batch size", "一次送進模型的一小批圖片 / 這批有幾張"],
        ["loss", "模型答案和正確答案差多遠的數值，訓練就是讓它越來越小"],
        ["learning rate（lr）", "每次更新參數要走多大一步"],
        ["CNN", "卷積神經網路，擅長從圖片抽取邊緣、筆畫等局部特徵"],
        ["LSTM", "一種循環神經網路，擅長依序讀取序列，並記住前面看過的內容"],
        ["CTC", "讓序列輸出不需要逐字對齊標註的 loss 與解碼方法"],
        ["blank", "CTC 額外加的「這一步不輸出任何字」類別"],
        ["資料增強", "對訓練圖片做隨機變化，讓模型學到更通用的特徵"],
        ["驗證集", "訓練時不拿來學、只拿來打分數的資料，用來判斷模型有沒有真的學會"],
        ["過擬合", "模型把訓練資料背起來，對新資料卻表現不好"],
        ["checkpoint", "存下來的模型權重檔（這裡是 plate_crnn.pt）"],
    ], [0.27, 0.73])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Doc(OUT).multiBuild(story)
    print("OK", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="產生專案教學 PDF")
    ap.add_argument("--run", default="", help="訓練曲線要用哪次訓練（models/runs/ 底下的資料夾名稱）")
    build(pick_run(ap.parse_args().run))
