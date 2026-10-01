const $ = (id) => document.getElementById(id);

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------- 模型狀態 ----------
let datasetExists = false;

async function refreshStatus() {
  const s = await getJSON("/api/status");
  datasetExists = "train" in s.dataset && "val" in s.dataset;
  $("datasetInfo").textContent = datasetExists
    ? `資料集已存在（train ${s.dataset.train} 張、val ${s.dataset.val} 張），直接訓練即可`
    : "尚未產生資料集";
  $("modelStatus").textContent = s.model_ready
    ? `使用模型 ${s.active}（第 ${s.epoch} epoch，驗證正確率 ${pct(s.val_seq_acc || 0)}）`
    : "尚未訓練模型：請執行 python train.py";
}

// ---------- 上傳 + 框選 ----------
const photo = $("photo");
const pctx = photo.getContext("2d");
let image = null;       // 原始圖片
let box = null;         // 框選範圍（原圖座標）
let dragStart = null;

$("fileInput").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    image = img;
    box = null;
    photo.width = img.naturalWidth;
    photo.height = img.naturalHeight;
    drawPhoto();
    updateCropPreview();
    $("recognizeBtn").disabled = false;
    $("clearBoxBtn").disabled = false;
  };
  img.src = URL.createObjectURL(file);
};

function drawPhoto() {
  pctx.drawImage(image, 0, 0);
  if (box) {
    pctx.fillStyle = "rgba(0,0,0,.45)";
    pctx.fillRect(0, 0, photo.width, photo.height);
    pctx.drawImage(image, box.x, box.y, box.w, box.h, box.x, box.y, box.w, box.h);
    pctx.strokeStyle = "#22c55e";
    pctx.lineWidth = Math.max(2, photo.width / 300);
    pctx.strokeRect(box.x, box.y, box.w, box.h);
  }
}

function toImageCoords(e) {
  const r = photo.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * photo.width,
    y: ((e.clientY - r.top) / r.height) * photo.height,
  };
}

photo.addEventListener("pointerdown", (e) => {
  if (!image) return;
  dragStart = toImageCoords(e);
  photo.setPointerCapture(e.pointerId);
});
photo.addEventListener("pointermove", (e) => {
  if (!dragStart) return;
  const p = toImageCoords(e);
  box = {
    x: Math.max(0, Math.min(dragStart.x, p.x)),
    y: Math.max(0, Math.min(dragStart.y, p.y)),
    w: Math.abs(p.x - dragStart.x),
    h: Math.abs(p.y - dragStart.y),
  };
  drawPhoto();
});
photo.addEventListener("pointerup", () => {
  dragStart = null;
  if (box && (box.w < 5 || box.h < 5)) box = null;
  drawPhoto();
  updateCropPreview();
});

$("clearBoxBtn").onclick = () => {
  box = null;
  drawPhoto();
  updateCropPreview();
};

function cropCanvas() {
  const r = box || { x: 0, y: 0, w: image.naturalWidth, h: image.naturalHeight };
  const c = document.createElement("canvas");
  c.width = Math.round(r.w);
  c.height = Math.round(r.h);
  c.getContext("2d").drawImage(image, r.x, r.y, r.w, r.h, 0, 0, c.width, c.height);
  return c;
}

function updateCropPreview() {
  const prev = $("cropPreview");
  const ctx = prev.getContext("2d");
  ctx.clearRect(0, 0, prev.width, prev.height);
  if (image) ctx.drawImage(cropCanvas(), 0, 0, prev.width, prev.height);
}

$("recognizeBtn").onclick = async () => {
  const blob = await new Promise((r) => cropCanvas().toBlob(r, "image/png"));
  const form = new FormData();
  form.append("image", blob, "crop.png");
  $("uploadText").textContent = "…";
  try {
    const r = await getJSON("/api/recognize", { method: "POST", body: form });
    $("uploadText").textContent = r.text || "（未辨識出文字）";
    $("uploadConf").textContent = `信心度 ${pct(r.confidence)}`;
  } catch (e) {
    $("uploadText").textContent = "—";
    $("uploadConf").textContent = e.message;
  }
};

// ---------- 隨機測試 ----------
$("randomBtn").onclick = async () => {
  try {
    const r = await getJSON("/api/random");
    $("randomResult").hidden = false;
    $("randomImg").src = r.image;
    $("randomTruth").textContent = r.truth;
    $("randomText").textContent = r.text || "（空）";
    const ok = r.text === r.truth;
    $("randomMark").textContent = ok ? "✔ 正確" : "✘ 錯誤";
    $("randomMark").className = ok ? "ok-mark" : "bad-mark";
    $("randomConf").textContent = `信心度 ${pct(r.confidence)}`;
  } catch (e) {
    $("randomResult").hidden = false;
    $("randomTruth").textContent = "";
    $("randomText").textContent = e.message;
  }
};

// ---------- 驗證集抽樣 ----------
async function loadValSamples() {
  const grid = $("valGrid");
  try {
    const items = await getJSON("/api/val-samples?n=24");
    const correct = items.filter((s) => s.text === s.truth).length;
    $("valSummary").textContent = `本次抽樣 ${items.length} 張，正確 ${correct} 張（${pct(correct / items.length)}）`;
    grid.innerHTML = "";
    for (const s of items) {
      const ok = s.text === s.truth;
      const div = document.createElement("div");
      div.className = `sample ${ok ? "ok" : "bad"}`;
      div.innerHTML = `<img><div class="pred"></div><div class="hint"></div>`;
      div.querySelector("img").src = s.url;
      div.querySelector(".pred").textContent = `${ok ? "✔" : "✘"} ${s.text || "（空）"}`;
      div.querySelector(".hint").textContent = `答案 ${s.truth} · ${pct(s.confidence)}`;
      grid.appendChild(div);
    }
  } catch (e) {
    $("valSummary").textContent = e.message;
  }
}
$("valBtn").onclick = loadValSamples;

// ---------- 訓練曲線 ----------
let chart = null;
async function loadHistory() {
  const h = await getJSON("/api/history");
  if (h.length === 0) {
    $("historyInfo").textContent = "尚無訓練紀錄";
    return;
  }
  const last = h[h.length - 1];
  $("historyInfo").textContent =
    `共 ${h.length} 個 epoch，最後：loss ${last.train_loss}、整張正確率 ${pct(last.val_seq_acc)}、字元正確率 ${pct(last.val_char_acc)}`;
  const data = {
    labels: h.map((e) => e.epoch),
    datasets: [
      { label: "訓練 loss", data: h.map((e) => e.train_loss), yAxisID: "loss", borderColor: "#f97316", backgroundColor: "#f97316" },
      { label: "驗證整張正確率", data: h.map((e) => e.val_seq_acc), yAxisID: "acc", borderColor: "#3b82f6", backgroundColor: "#3b82f6" },
      { label: "驗證字元正確率", data: h.map((e) => e.val_char_acc), yAxisID: "acc", borderColor: "#22c55e", backgroundColor: "#22c55e" },
    ],
  };
  if (chart) {
    chart.data = data;
    chart.update();
    return;
  }
  chart = new Chart($("historyChart"), {
    type: "line",
    data,
    options: {
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "epoch" } },
        loss: { type: "linear", position: "left", title: { display: true, text: "loss" } },
        acc: { type: "linear", position: "right", min: 0, max: 1, grid: { drawOnChartArea: false },
               ticks: { callback: (v) => `${v * 100}%` } },
      },
    },
  });
}

refreshStatus();
loadHistory();
loadValSamples();
// 訓練時開著網頁，每 10 秒更新一次曲線與狀態
setInterval(() => { loadHistory(); refreshStatus(); }, 10000);

// ---------- 訓練工作台 ----------
let wasRunning = false;

async function startJob(url, body) {
  try {
    await getJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    pollJob();
  } catch (e) {
    $("jobState").textContent = `· ${e.message}`;
  }
}

$("genBtn").onclick = () =>
  startJob("/api/jobs/generate", { train: $("genTrain").value, val: $("genVal").value });

$("trainBtn").onclick = () =>
  startJob("/api/jobs/train", {
    epochs: $("trEpochs").value,
    batch: $("trBatch").value,
    lr: $("trLr").value,
    name: $("trName").value,
  });

$("stopBtn").onclick = () => getJSON("/api/jobs/stop", { method: "POST" }).then(pollJob);

async function pollJob() {
  const j = await getJSON("/api/jobs");
  const log = $("jobLog");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 10;
  log.textContent = j.log || "尚未執行";
  if (atBottom) log.scrollTop = log.scrollHeight;

  $("genBtn").disabled = j.running || datasetExists;
  $("trainBtn").disabled = j.running;
  $("stopBtn").disabled = !j.running;
  if (j.running) {
    $("jobState").textContent = `· ${j.name} 執行中…`;
  } else if (j.name) {
    $("jobState").textContent = j.returncode === 0 ? `· ${j.name} 完成` : `· ${j.name} 已結束（代碼 ${j.returncode}）`;
  }

  if (wasRunning && !j.running) {
    // 工作剛結束：更新模型狀態、曲線、驗證集結果
    refreshStatus();
    loadHistory();
    loadValSamples();
    loadModels();
  }
  if (j.running && j.name === "訓練模型") {
    loadHistory();
    loadModels();
  }
  wasRunning = j.running;
}

pollJob();
setInterval(pollJob, 2000);

// ---------- 訓練進度條 ----------
const clock = (sec) => {
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mmss = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return h ? `${h}:${mmss}` : mmss;
};
const setBar = (id, frac) => { $(id).style.width = `${Math.min(100, frac * 100).toFixed(1)}%`; };

async function loadProgress() {
  let p;
  try {
    p = await getJSON("/api/progress");
  } catch (e) {
    return; // 檔案剛好在寫入，等下一次
  }
  const box = $("progressBox");
  if (!p) { box.hidden = true; return; }
  box.hidden = false;

  const stale = p.stage !== "done" && p.age > 60; // 超過 1 分鐘沒更新：被停止或當掉
  box.className = `progress-box ${stale ? "stopped" : p.stage === "done" ? "" : "running"}`;
  const epochFrac = p.batches ? p.batch / p.batches : 0;
  const total = p.stage === "done" ? 1 : p.epoch ? (p.epoch - 1 + epochFrac) / p.epochs : 0;

  let title;
  if (stale) title = "訓練已停止或中斷";
  else if (p.stage === "loading") title = `載入資料中（${p.split}）…`;
  else if (p.stage === "evaluating") title = `Epoch ${p.epoch}/${p.epochs} · 驗證中…`;
  else if (p.stage === "done") title = "訓練完成 ✔";
  else title = `訓練中 · Epoch ${p.epoch}/${p.epochs}`;
  $("progressTitle").textContent = title;
  $("progressPct").textContent = pct(total);
  setBar("progressBar", total);

  if (p.stage === "loading") {
    $("progressSub").textContent = `已載入 ${p.done} / ${p.total} 張`;
    $("progressEpochPct").textContent = pct(p.done / p.total);
    setBar("progressEpochBar", p.done / p.total);
  } else {
    $("progressSub").textContent = p.batches ? `本 epoch：batch ${p.batch} / ${p.batches}` : "";
    $("progressEpochPct").textContent = p.batches ? pct(epochFrac) : "";
    setBar("progressEpochBar", epochFrac);
  }

  const parts = [p.run];
  if (p.elapsed != null) parts.push(`已經過 ${clock(p.elapsed)}`);
  if (p.eta != null && p.stage !== "done" && !stale) parts.push(`預估剩餘 ${clock(p.eta)}`);
  parts.push(`${Math.round(p.age)} 秒前更新`);
  $("progressTime").textContent = parts.join(" · ");
}

loadProgress();
setInterval(loadProgress, 2000);

// ---------- 已儲存的模型 ----------
async function loadModels() {
  const { active, runs } = await getJSON("/api/models");
  const tbody = $("modelTable");
  if (runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="hint">尚無模型，請先訓練</td></tr>`;
    return;
  }
  tbody.innerHTML = "";
  for (const r of runs) {
    const tr = document.createElement("tr");
    if (r.name === active) tr.className = "active-run";
    const cells = [
      r.name, r.epochs ?? "—", r.batch ?? "—", r.lr ?? "—", r.train_samples ?? "—",
      r.best_val_seq_acc == null ? "—" : `${pct(r.best_val_seq_acc)}（第 ${r.best_epoch} epoch）`,
      r.best_val_char_acc == null ? "—" : pct(r.best_val_char_acc),
      r.finished ? "完成" : "訓練中 / 中斷",
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    const btn = document.createElement("button");
    if (r.name === active) {
      btn.textContent = "使用中";
      btn.disabled = true;
    } else {
      btn.textContent = "使用";
      btn.onclick = async () => {
        await getJSON("/api/models/active", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run: r.name }),
        });
        refreshStatus();
        loadModels();
        loadHistory();
        loadValSamples();
      };
    }
    td.appendChild(btn);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}
loadModels();
