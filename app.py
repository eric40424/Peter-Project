"""車牌辨識網站 - Flask 後端。

執行：python app.py，然後開啟 http://127.0.0.1:5000
訓練中也可以開著網站，模型檔更新後會自動重新載入。
每次訓練的模型都存在 models/runs/ 底下，可在網頁上切換要使用哪一個。
"""
import base64
import csv
import io
import json
import os
import random
import subprocess
import sys
import threading
import time

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from generate_dataset import available_fonts, make_sample
from plate_model import greedy_decode, load_model, preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(MODELS_DIR, "runs")
ACTIVE_PATH = os.path.join(MODELS_DIR, "active.txt")
LOG_PATH = os.path.join(BASE_DIR, "logs", "job.log")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # 不讓瀏覽器快取 app.js/style.css，改完程式重新整理就生效
lock = threading.Lock()
_model, _model_key = None, None
_fonts = None


def run_file(run, name):
    return os.path.join(RUNS_DIR, run, name)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def active_run():
    """網站目前使用的模型名稱（models/active.txt）。"""
    if os.path.exists(ACTIVE_PATH):
        with open(ACTIVE_PATH, encoding="utf-8") as f:
            run = f.read().strip()
        if os.path.exists(run_file(run, "plate_crnn.pt")):
            return run
    return None


def get_model():
    """載入目前使用的模型；切換模型或模型檔更新（例如訓練中）時重新載入。"""
    global _model, _model_key
    run = active_run()
    if run is None:
        return None
    path = run_file(run, "plate_crnn.pt")
    key = (run, os.path.getmtime(path))
    if key != _model_key:
        _model, _model_key = load_model(path), key
    return _model


def recognize(images):
    with lock:
        model = get_model()
        if model is None:
            return None
        with torch.no_grad():
            return greedy_decode(model(torch.stack([preprocess(im) for im in images])))


def no_model():
    return jsonify(error="尚未有訓練好的模型，請先執行 python train.py"), 400


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    run = active_run()
    info = {"model_ready": run is not None, "active": run, "dataset": {}}
    for split in ("train", "val"):
        path = os.path.join(DATASET_DIR, split, "labels.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                info["dataset"][split] = sum(1 for _ in f) - 1
    if run:
        cfg = read_json(run_file(run, "config.json"), {})
        info.update(epoch=cfg.get("best_epoch"), val_seq_acc=cfg.get("best_val_seq_acc"))
    return jsonify(info)


@app.route("/api/models")
def api_models():
    """列出所有已儲存的模型（新的在前）。"""
    runs = []
    if os.path.isdir(RUNS_DIR):
        for run in sorted(os.listdir(RUNS_DIR), reverse=True):
            if os.path.exists(run_file(run, "plate_crnn.pt")):
                cfg = read_json(run_file(run, "config.json"), {"name": run})
                cfg["name"] = run
                runs.append(cfg)
    return jsonify(active=active_run(), runs=runs)


@app.route("/api/models/active", methods=["POST"])
def api_set_active():
    run = str(request.get_json(force=True).get("run", ""))
    if os.path.basename(run) != run or not os.path.exists(run_file(run, "plate_crnn.pt")):
        return jsonify(error="找不到這個模型"), 404
    with open(ACTIVE_PATH, "w", encoding="utf-8") as f:
        f.write(run)
    return jsonify(active=run)


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="請上傳圖片"), 400
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify(error="無法讀取圖片"), 400
    result = recognize([img])
    if result is None:
        return no_model()
    text, conf = result[0]
    return jsonify(text=text, confidence=conf)


@app.route("/api/random")
def api_random():
    """產生一張全新的合成車牌並辨識。"""
    global _fonts
    _fonts = _fonts or available_fonts()
    img, truth = make_sample(_fonts)
    result = recognize([img])
    if result is None:
        return no_model()
    text, conf = result[0]
    return jsonify(image=to_data_url(img), truth=truth, text=text, confidence=conf)


@app.route("/api/val-samples")
def api_val_samples():
    """從驗證集隨機抽樣並辨識，用來快速檢視模型效果。"""
    labels_path = os.path.join(DATASET_DIR, "val", "labels.csv")
    if not os.path.exists(labels_path):
        return jsonify(error="找不到驗證集，請先執行 python generate_dataset.py"), 400
    with open(labels_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = min(int(request.args.get("n", 24)), 100, len(rows))
    rows = random.sample(rows, n)
    images = [Image.open(os.path.join(DATASET_DIR, "val", r["filename"])).convert("RGB") for r in rows]
    result = recognize(images)
    if result is None:
        return no_model()
    return jsonify([
        {"url": f"/dataset/val/{r['filename']}", "truth": r["text"], "text": t, "confidence": c}
        for r, (t, c) in zip(rows, result)
    ])


@app.route("/api/history")
def api_history():
    run = request.args.get("run") or active_run()
    if not run or os.path.basename(run) != run:
        return jsonify([])
    return jsonify(read_json(run_file(run, "history.json"), []))


# ---------- 從網頁啟動「產生資料集」/「訓練」 ----------
job = {"name": None, "proc": None}


def job_running():
    return job["proc"] is not None and job["proc"].poll() is None


def start_job(name, args):
    if job_running():
        return jsonify(error=f"目前正在執行「{job['name']}」，請等它結束或先停止"), 409
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log = open(LOG_PATH, "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    job["proc"] = subprocess.Popen(
        [sys.executable, "-u"] + args, cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT, env=env
    )
    job["name"] = name
    return jsonify(ok=True)


def int_arg(body, key, default, lo, hi):
    try:
        return str(max(lo, min(hi, int(body.get(key, default)))))
    except (TypeError, ValueError):
        return str(default)


@app.route("/api/jobs/generate", methods=["POST"])
def job_generate():
    b = request.get_json(force=True)
    return start_job("產生資料集", [
        "generate_dataset.py",
        "--train", int_arg(b, "train", 30000, 100, 200000),
        "--val", int_arg(b, "val", 3000, 50, 20000),
    ])


@app.route("/api/jobs/train", methods=["POST"])
def job_train():
    b = request.get_json(force=True)
    try:
        lr = str(max(1e-5, min(1e-1, float(b.get("lr", 1e-3)))))
    except (TypeError, ValueError):
        lr = "0.001"
    return start_job("訓練模型", [
        "train.py",
        "--epochs", int_arg(b, "epochs", 30, 1, 500),
        "--batch", int_arg(b, "batch", 128, 8, 1024),
        "--lr", lr,
        "--name", str(b.get("name", ""))[:40],
    ])


@app.route("/api/jobs/stop", methods=["POST"])
def job_stop():
    if job_running():
        pid = job["proc"].pid
        if os.name == "nt":  # 連同子行程（多行程產生資料）一起結束
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            job["proc"].terminate()
    return jsonify(ok=True)


@app.route("/api/jobs")
def job_status():
    lines = []
    if os.path.exists(LOG_PATH):
        # newline="" 保留原始的 \r（預設會被轉成換行，進度條就會一行疊一行）
        with open(LOG_PATH, encoding="utf-8", errors="replace", newline="") as f:
            # 進度列用 \r 覆寫，只保留每行最後的狀態（先去掉 Windows 換行 \r\n 的 \r）
            lines = [l.rstrip("\r").split("\r")[-1] for l in f.read().split("\n")]
    proc = job["proc"]
    return jsonify(
        name=job["name"],
        running=job_running(),
        returncode=None if proc is None else proc.poll(),
        log="\n".join(lines[-40:]),
    )


@app.route("/api/progress")
def api_progress():
    """最近一次訓練的進度（train.py 寫的 progress.json），網頁用來畫進度條。"""
    paths = [run_file(r, "progress.json") for r in os.listdir(RUNS_DIR)] if os.path.isdir(RUNS_DIR) else []
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return jsonify(None)
    latest = max(paths, key=os.path.getmtime)
    try:
        p = read_json(latest, None) or {}
    except (OSError, ValueError):  # train.py 剛好在覆寫檔案，下次輪詢再讀
        return jsonify(error="progress.json 暫時無法讀取"), 503
    p["run"] = os.path.basename(os.path.dirname(latest))
    p["age"] = max(0.0, time.time() - p.get("updated", 0))  # 幾秒前更新，太久沒更新代表被停止
    return jsonify(p)


@app.route("/dataset/<split>/<filename>")
def dataset_file(split, filename):
    if split not in ("train", "val"):
        return "", 404
    return send_from_directory(os.path.join(DATASET_DIR, split), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
