// Demo "Coba Model": memuat model ONNX di browser dan mengklasifikasikan foto sampah.
import { CATEGORIES } from "./categories.js";
import { preprocess } from "./preprocess.js";

const ORT_VERSION = "1.30.0";
const MODEL_URL = "models/garbage_classifier.onnx";
const LOW_CONFIDENCE = 0.5;
const MAX_PIXELS = 16_000_000; // foto HP yang sangat besar diperkecil dulu supaya tidak kehabisan memori

const $ = (id) => document.getElementById(id);
const ms = (v) => (v < 1 ? "<1 ms" : `${Math.round(v)} ms`);
const pct = (p) => `${(p * 100).toLocaleString("id-ID", { maximumFractionDigits: 1, minimumFractionDigits: 1 })}%`;

const state = { session: null, config: null, labels: null, samples: [], busy: false, onTiming: null };

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal memuat ${url} (${res.status})`);
  return res.json();
}

// Unduh model sambil melaporkan progres (file sekitar 8,7 MB)
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Gagal memuat model (${res.status})`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !total) {
    const buf = await res.arrayBuffer();
    onProgress(1);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(Math.min(loaded / total, 1));
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

function setStatus(kind, text, progress) {
  const el = $("model-status");
  el.classList.toggle("ready", kind === "ready");
  el.classList.toggle("error", kind === "error");
  $("model-status-text").textContent = text;
  if (progress !== undefined) $("model-progress").style.width = `${Math.round(progress * 100)}%`;
}

function setInputsEnabled(enabled) {
  for (const el of document.querySelectorAll("[data-needs-model]")) el.disabled = !enabled;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Decode gambar tanpa konversi profil warna (PIL juga mengabaikannya), orientasi EXIF diterapkan
async function decode(blob) {
  const bitmap = await createImageBitmap(blob, {
    imageOrientation: "from-image",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  let { width, height } = bitmap;
  if (width * height > MAX_PIXELS) {
    const s = Math.sqrt(MAX_PIXELS / (width * height));
    width = Math.round(width * s);
    height = Math.round(height * s);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return ctx.getImageData(0, 0, width, height);
}

async function classify(blob) {
  const t0 = performance.now();
  const imageData = await decode(blob);
  const input = preprocess(imageData, state.config);
  const t1 = performance.now();
  const tensor = new ort.Tensor("float32", input, state.config.input_shape);
  const output = await state.session.run({ [state.config.input_name]: tensor });
  const t2 = performance.now();
  const probs = softmax(Array.from(output[state.config.output_name].data));
  const ranked = probs.map((p, i) => ({ label: state.labels[i], p })).sort((a, b) => b.p - a.p);
  return { ranked, probs, preprocessMs: t1 - t0, inferenceMs: t2 - t1 };
}

function showPreview(blob, alt) {
  const zone = $("dropzone");
  const url = URL.createObjectURL(blob);
  let img = zone.querySelector("img");
  if (!img) {
    img = document.createElement("img");
    zone.prepend(img);
  }
  const old = img.dataset.url;
  img.src = url;
  img.alt = alt;
  img.dataset.url = url;
  if (old) URL.revokeObjectURL(old);
  zone.querySelector(".dropzone-empty")?.remove();
}

function renderResult({ ranked, preprocessMs, inferenceMs }, sample) {
  const top = ranked[0];
  const cat = CATEGORIES[top.label];
  const low = top.p < LOW_CONFIDENCE;
  const truth = sample
    ? `<div class="truth">Label asli di dataset: <strong>${CATEGORIES[sample.label].name}</strong>
        ${sample.label === top.label
          ? '<span class="badge ok">Tebakan benar</span>'
          : '<span class="badge no">Tebakan salah</span>'}</div>`
    : "";
  const warn = low
    ? `<div class="alert" role="status"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17.5v.5"/></svg>
        <div><strong>Model kurang yakin.</strong> Confidence tertinggi di bawah 50%. Coba foto ulang dengan satu objek, pencahayaan cukup, dan latar polos.</div></div>`
    : "";
  const bars = ranked.slice(0, 3).map((r) => {
    const c = CATEGORIES[r.label];
    return `<div class="bar-row" style="--cat: var(--c-${r.label})">
        <span class="mini-icon">${c.icon}</span>
        <span class="name">${c.name}</span>
        <span class="bar-track" role="progressbar" aria-label="${c.name}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(r.p * 100)}"><span class="bar-fill" style="width: ${(r.p * 100).toFixed(2)}%"></span></span>
        <span class="pct">${pct(r.p)}</span>
      </div>`;
  }).join("");

  $("result").innerHTML = `
    <div class="top1" style="--cat: var(--c-${top.label})">
      <span class="cat-icon">${cat.icon}</span>
      <div>
        <div class="top1-label">Prediksi teratas</div>
        <div class="top1-name">${cat.name} <small>${cat.en}</small></div>
      </div>
      <div class="top1-conf"><div class="v">${pct(top.p)}</div><div class="l">confidence</div></div>
    </div>
    ${warn}
    ${truth}
    <div class="bars">${bars}</div>
    <div class="cat-info">
      <div><h4>Contoh barang</h4><ul>${cat.examples.map((e) => `<li>${e}</li>`).join("")}</ul></div>
      <div><h4>Cara membuang yang benar</h4><ul>${cat.tips.map((e) => `<li>${e}</li>`).join("")}</ul></div>
      <div class="bin"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M9 6V4h6v2M6 6l1 15h10l1-15"/></svg><span>${cat.bin}</span></div>
    </div>
    <div class="timing">Diproses di perangkat Anda: preprocessing ${ms(preprocessMs)}, inference ${ms(inferenceMs)}. Foto tidak dikirim ke server mana pun.</div>`;
}

async function run(blob, { alt = "Foto yang diunggah", sample = null } = {}) {
  if (!state.session || state.busy) return;
  state.busy = true;
  setInputsEnabled(false);
  showPreview(blob, alt);
  $("processing").classList.add("show");
  try {
    const result = await classify(blob);
    renderResult(result, sample);
    state.onTiming?.(result.inferenceMs);
    return result;
  } catch (err) {
    console.error(err);
    $("result").innerHTML = `<div class="alert" role="alert"><div><strong>Gambar tidak bisa diproses.</strong> Pastikan file berupa foto (JPG, PNG, atau WebP).</div></div>`;
  } finally {
    $("processing").classList.remove("show");
    state.busy = false;
    setInputsEnabled(true);
  }
}

function markSample(button) {
  for (const b of document.querySelectorAll(".sample")) b.setAttribute("aria-pressed", String(b === button));
}

function renderSamples() {
  const wrap = $("samples");
  wrap.innerHTML = state.samples.map((s, i) => `
    <button class="sample" type="button" data-index="${i}" data-needs-model disabled aria-pressed="false"
      aria-label="Coba contoh ${CATEGORIES[s.label].name} ${s.file.match(/_(\d+)/)[1]}">
      <img src="assets/samples/${s.file}" alt="" loading="lazy" width="${s.width}" height="${s.height}">
    </button>`).join("");
  wrap.addEventListener("click", async (e) => {
    const btn = e.target.closest(".sample");
    if (!btn || btn.disabled) return;
    const s = state.samples[Number(btn.dataset.index)];
    markSample(btn);
    const blob = await (await fetch(`assets/samples/${s.file}`)).blob();
    run(blob, { alt: `Contoh gambar ${CATEGORIES[s.label].name}`, sample: s });
  });
}

function bindInputs() {
  const onFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      $("result").innerHTML = `<div class="alert" role="alert"><div><strong>File bukan gambar.</strong> Pilih file JPG, PNG, atau WebP.</div></div>`;
      return;
    }
    markSample(null);
    run(file, { alt: "Foto yang diunggah" });
  };
  $("file-input").addEventListener("change", (e) => { onFile(e.target.files[0]); e.target.value = ""; });
  $("camera-input").addEventListener("change", (e) => { onFile(e.target.files[0]); e.target.value = ""; });
  $("btn-upload").addEventListener("click", () => $("file-input").click());
  $("btn-camera").addEventListener("click", () => $("camera-input").click());

  const zone = $("dropzone");
  zone.addEventListener("click", (e) => {
    if (state.session && !state.busy && !e.target.closest("img")) $("file-input").click();
  });
  zone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && state.session) { e.preventDefault(); $("file-input").click(); }
  });
  for (const ev of ["dragenter", "dragover"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); });
  }
  zone.addEventListener("drop", (e) => onFile(e.dataTransfer?.files?.[0]));
  // Tempel gambar dari clipboard (Ctrl+V)
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) onFile(item.getAsFile());
  });
}

async function loadModel() {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  // Multi-thread butuh cross-origin isolation yang tidak tersedia di GitHub Pages
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  setStatus("loading", "Mengunduh model (8,7 MB)...", 0);
  const buffer = await fetchWithProgress(MODEL_URL, (p) => setStatus("loading", `Mengunduh model... ${Math.round(p * 100)}%`, p));
  setStatus("loading", "Menyiapkan model...", 1);
  state.session = await ort.InferenceSession.create(buffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  // Pemanasan supaya prediksi pertama pengunjung tidak lambat
  const warm = new ort.Tensor("float32", new Float32Array(state.config.input_shape.reduce((a, b) => a * b, 1)), state.config.input_shape);
  await state.session.run({ [state.config.input_name]: warm });
}

export async function initDemo({ onTiming } = {}) {
  state.onTiming = onTiming;
  bindInputs();
  try {
    if (typeof ort === "undefined") throw new Error("onnxruntime-web gagal dimuat dari CDN");
    [state.config, state.labels, state.samples] = await Promise.all([
      fetchJSON("models/preprocessing.json"),
      fetchJSON("models/labels.json"),
      fetchJSON("assets/samples/samples.json"),
    ]);
    renderSamples();
    await loadModel();
    setStatus("ready", "Model siap, berjalan di perangkat Anda");
    setInputsEnabled(true);
    // Tautan langsung ke contoh tertentu, misalnya ?sample=paper_2.jpg
    const requested = new URLSearchParams(location.search).get("sample");
    const idx = state.samples.findIndex((s) => s.file === requested);
    if (idx >= 0) document.querySelector(`.sample[data-index="${idx}"]`)?.click();
  } catch (err) {
    console.error(err);
    setStatus("error", "Model gagal dimuat. Periksa koneksi lalu muat ulang halaman.");
  }
  document.dispatchEvent(new Event("modelready"));

  // Alat cek: bandingkan prediksi browser dengan prediksi notebook untuk ke-12 contoh
  window.garbageDemo = {
    async selfTest() {
      const rows = [];
      for (const s of state.samples) {
        const blob = await (await fetch(`assets/samples/${s.file}`)).blob();
        const { ranked } = await classify(blob);
        rows.push({
          file: s.file,
          label: s.label,
          expected: s.expected_prediction,
          browser: ranked[0].label,
          expectedConf: s.expected_confidence,
          browserConf: Number(ranked[0].p.toFixed(4)),
          match: ranked[0].label === s.expected_prediction && Math.abs(ranked[0].p - s.expected_confidence) < 0.005,
        });
      }
      console.table(rows);
      return rows;
    },
  };
}
