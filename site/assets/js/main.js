// Inisialisasi halaman: tema, KPI, tabel, demo, dan chart (Plotly dimuat saat dibutuhkan).
import { initDemo } from "./demo.js";
import { initCharts } from "./charts.js";

const PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js";
const MODEL_KEYS = ["A", "B", "C"];
const fmt = (v, digits = 1) => v.toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ tema
function currentTheme() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  $("theme-toggle").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* penyimpanan tidak tersedia */ }
    document.dispatchEvent(new Event("themechange"));
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.dataset.theme) document.dispatchEvent(new Event("themechange"));
  });
}

// ------------------------------------------------------------ data
async function loadData() {
  const files = {
    comparison: "data/model_comparison.json",
    history: "data/training_history.json",
    confusion: "data/confusion_matrix.json",
    perClass: "data/per_class_metrics.json",
    robustness: "data/robustness.json",
    dataset: "data/dataset_summary.json",
  };
  const entries = await Promise.all(Object.entries(files).map(async ([k, url]) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Gagal memuat ${url}`);
    return [k, await res.json()];
  }));
  return Object.fromEntries(entries);
}

function fillKpis(data) {
  const best = data.comparison.models[data.comparison.selected_model.key];
  $("kpi-acc").textContent = `${fmt(best.accuracy * 100)}%`;
  $("kpi-f1").textContent = fmt(best.macro_f1, 3);
  $("kpi-size").textContent = `${fmt(best.size_mb)} MB`;
  $("kpi-speed").textContent = `±${Math.round(best.cpu_ms)} ms`;
}

function fillModelTable(data) {
  const m = data.comparison.models;
  const selected = data.comparison.selected_model.key;
  $("model-table-body").innerHTML = MODEL_KEYS.map((k) => `
    <tr class="${k === selected ? "selected" : ""}">
      <td><span class="swatch" style="display:inline-block;background:var(--m-${k.toLowerCase()});margin-right:8px"></span>${k}. ${m[k].model}${k === selected ? ' <span class="chip">terpilih</span>' : ""}</td>
      <td class="n">${fmt(m[k].accuracy * 100)}%</td>
      <td class="n">${fmt(m[k].macro_f1, 3)}</td>
      <td class="n">${(m[k].total_params / 1e6).toLocaleString("id-ID", { maximumFractionDigits: 1 })} jt</td>
      <td class="n">${fmt(m[k].size_mb)} MB</td>
      <td class="n">${fmt(m[k].gpu_ms)} ms</td>
      <td class="n">${fmt(m[k].cpu_ms)} ms</td>
    </tr>`).join("");
}

// Waktu inference di browser pengunjung menggantikan angka CPU laptop di KPI
function makeTimingReporter() {
  const times = [];
  return (ms) => {
    times.push(ms);
    const sorted = [...times].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    $("kpi-speed").textContent = median < 1 ? "<1 ms" : `${Math.round(median)} ms`;
    $("kpi-speed-note").textContent = `di perangkat Anda (${times.length} prediksi)`;
  };
}

// ------------------------------------------------------------ Plotly dimuat saat pengunjung mendekati bagian chart
function loadPlotly() {
  return new Promise((resolve, reject) => {
    if (typeof Plotly !== "undefined") return resolve();
    const s = document.createElement("script");
    s.src = PLOTLY_URL;
    s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Plotly gagal dimuat"));
    document.head.appendChild(s);
  });
}

function whenNear(el, callback) {
  if (!("IntersectionObserver" in window)) return callback();
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      callback();
    }
  }, { rootMargin: "900px 0px" });
  io.observe(el);
}

async function main() {
  initTheme();
  $("year").textContent = new Date().getFullYear();

  const dataPromise = loadData();
  let started = false;
  const startCharts = async () => {
    if (started) return;
    started = true;
    try {
      const [data] = await Promise.all([dataPromise, loadPlotly()]);
      initCharts(data);
    } catch (err) {
      console.error(err);
      for (const el of document.querySelectorAll(".chart-loading")) el.textContent = "Chart gagal dimuat. Periksa koneksi internet.";
    }
  };
  // Muat saat pengunjung mendekati bagian chart, atau setelah model demo siap (mana yang lebih dulu),
  // supaya unduhan Plotly tidak berebut bandwidth dengan model di awal
  whenNear($("dataset"), startCharts);
  document.addEventListener("modelready", startCharts, { once: true });

  initDemo({ onTiming: makeTimingReporter() });

  try {
    const data = await dataPromise;
    fillKpis(data);
    fillModelTable(data);
  } catch (err) {
    console.error(err);
  }
}

main();
