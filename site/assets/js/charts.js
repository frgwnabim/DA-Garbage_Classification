// Chart interaktif (Plotly.js). Warna dibaca dari CSS variable supaya ikut tema terang/gelap.
import { CATEGORIES } from "./categories.js";

const MODEL_KEYS = ["A", "B", "C"];
const MODEL_NAMES = { A: "CNN dari nol", B: "MobileNetV2", C: "ResNet50" };
const CONFIG = { displayModeBar: false, responsive: true };
const state = { data: null, visibleModels: new Set(["A", "B", "C"]), cmMode: "normalized" };

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isDark = () => css("color-scheme") === "dark";
const catColor = (label) => css(`--c-${label}`);
const modelColor = (key) => css(`--m-${key.toLowerCase()}`);
const catName = (label) => CATEGORIES[label].name;

// Pilih teks hitam atau putih yang paling kontras di atas warna latar (luminans relatif WCAG)
function contrastText(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) ? "#0b0b0b" : "#ffffff";
}

// Garis referensi horizontal yang muncul di legend (bukan teks di dalam plot, supaya tidak menimpa bar)
const refLine = (y, name) => ({
  type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
  line: { color: css("--muted"), width: 1.5, dash: "dash" }, showlegend: true, name,
});

function baseLayout(extra = {}) {
  const text2 = css("--text-2");
  const grid = css("--grid");
  const axis = {
    gridcolor: grid, zerolinecolor: grid, linecolor: css("--border"),
    tickfont: { color: text2, size: 12 }, title: { font: { color: text2, size: 12 } },
    automargin: true,
  };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font") || "Plus Jakarta Sans, sans-serif", color: css("--text"), size: 13 },
    separators: ",.",
    margin: { t: 16, r: 12, b: 48, l: 56 },
    hoverlabel: {
      bgcolor: css("--surface"), bordercolor: css("--border"),
      font: { family: css("--font"), color: css("--text"), size: 13 },
    },
    legend: { orientation: "h", x: 0, y: 1.12, font: { color: text2, size: 12 }, bgcolor: "rgba(0,0,0,0)" },
    bargap: 0.28,
    bargroupgap: 0.06,
    xaxis: { ...axis, showgrid: false },
    yaxis: { ...axis },
    ...extra,
  };
}

function plot(id, traces, layout) {
  const el = document.getElementById(id);
  if (!el) return;
  el.querySelector(".chart-loading")?.remove();
  Plotly.react(el, traces, layout, CONFIG);
}

// ------------------------------------------------------------ dataset
function datasetChart() {
  const d = state.data.dataset;
  const classes = d.classes;
  plot("chart-dataset", [{
    type: "bar",
    x: classes.map(catName),
    y: classes.map((c) => d.per_class[c]),
    marker: { color: classes.map(catColor), line: { width: 0 } },
    text: classes.map((c) => d.per_class[c]),
    textposition: "outside",
    textfont: { color: css("--text-2") },
    cliponaxis: false,
    customdata: classes.map((c) => [d.per_split.train[c], d.per_split.val[c], d.per_split.test[c]]),
    hovertemplate: "<b>%{x}</b><br>Total %{y} gambar<br>Train %{customdata[0]} · Val %{customdata[1]} · Test %{customdata[2]}<extra></extra>",
  }], baseLayout({
    yaxis: { ...baseLayout().yaxis, title: { text: "Jumlah gambar", font: { color: css("--text-2"), size: 12 } }, rangemode: "tozero" },
    xaxis: { ...baseLayout().xaxis, title: { text: "Kelas", font: { color: css("--text-2"), size: 12 } } },
    margin: { t: 24, r: 12, b: 56, l: 60 },
  }));
}

// ------------------------------------------------------------ perbandingan model
function modelsChart() {
  const m = state.data.comparison.models;
  const metrics = [["accuracy", "Accuracy"], ["macro_f1", "Macro F1"], ["weighted_f1", "Weighted F1"]];
  const traces = MODEL_KEYS.map((k) => ({
    type: "bar",
    name: `${k}. ${MODEL_NAMES[k]}`,
    x: metrics.map(([, label]) => label),
    y: metrics.map(([key]) => m[k][key]),
    marker: { color: modelColor(k) },
    text: metrics.map(([key]) => m[k][key].toLocaleString("id-ID", { minimumFractionDigits: 3, maximumFractionDigits: 3 })),
    textposition: "outside",
    textfont: { color: css("--text-2"), size: 11 },
    cliponaxis: false,
    hovertemplate: `<b>${MODEL_NAMES[k]}</b><br>%{x}: %{y:.3f}<extra></extra>`,
  }));
  plot("chart-models", traces, baseLayout({
    barmode: "group",
    yaxis: { ...baseLayout().yaxis, range: [0, 1.08], title: { text: "Skor di test set (0 sampai 1)", font: { color: css("--text-2"), size: 12 } } },
    shapes: [refLine(0.85, "Target macro F1 0,85")],
    legend: { ...baseLayout().legend, y: 1.16 },
    margin: { t: 40, r: 12, b: 40, l: 60 },
  }));
}

// ------------------------------------------------------------ kurva training
function trainingCharts() {
  const exps = state.data.history.experiments;
  const fineTuneStart = exps.B.history.findIndex((h) => h.stage === "fine_tuning") + 0.5;
  for (const [id, key, label] of [["chart-loss", "loss", "Loss (weighted cross-entropy)"], ["chart-acc", "acc", "Accuracy"]]) {
    const traces = [];
    for (const k of MODEL_KEYS) {
      const h = exps[k].history;
      const x = h.map((_, i) => i + 1);
      const visible = state.visibleModels.has(k);
      traces.push({
        type: "scatter", mode: "lines", name: `${MODEL_NAMES[k]} (val)`, legendgroup: k, visible,
        x, y: h.map((e) => e[`val_${key}`]), line: { color: modelColor(k), width: 2.5 },
        hovertemplate: `<b>${MODEL_NAMES[k]}</b> val<br>Epoch %{x}: %{y:.3f}<extra></extra>`,
      });
      traces.push({
        type: "scatter", mode: "lines", name: `${MODEL_NAMES[k]} (train)`, legendgroup: k, visible,
        x, y: h.map((e) => e[`train_${key}`]), line: { color: modelColor(k), width: 1.8, dash: "dot" },
        hovertemplate: `<b>${MODEL_NAMES[k]}</b> train<br>Epoch %{x}: %{y:.3f}<extra></extra>`,
      });
    }
    const showFt = state.visibleModels.has("B") || state.visibleModels.has("C");
    plot(id, traces, baseLayout({
      showlegend: false,
      hovermode: "x unified",
      xaxis: { ...baseLayout().xaxis, title: { text: "Epoch", font: { color: css("--text-2"), size: 12 } }, showgrid: false },
      yaxis: { ...baseLayout().yaxis, title: { text: label, font: { color: css("--text-2"), size: 12 } } },
      shapes: showFt ? [{ type: "line", x0: fineTuneStart, x1: fineTuneStart, yref: "paper", y0: 0, y1: 1, line: { color: css("--muted"), width: 1, dash: "dash" } }] : [],
      annotations: showFt ? [{ x: fineTuneStart, yref: "paper", y: 1, xanchor: "left", yanchor: "top", text: " mulai fine-tuning (B, C)", showarrow: false, font: { color: css("--muted"), size: 11 } }] : [],
      margin: { t: 12, r: 12, b: 48, l: 60 },
    }));
  }
}

// ------------------------------------------------------------ confusion matrix
function confusionChart() {
  const cm = state.data.confusion;
  const normalized = state.cmMode === "normalized";
  const z = normalized ? cm.normalized : cm.matrix;
  const names = cm.classes.map(catName);
  const max = Math.max(...z.flat());
  const scale = isDark()
    ? [[0, "#16202a"], [0.5, "#256abf"], [1, "#b7d3f6"]]
    : [[0, "#f3f8fe"], [0.35, "#9ec5f4"], [0.7, "#2a78d6"], [1, "#0d366b"]];
  const annotations = [];
  z.forEach((row, i) => row.forEach((v, j) => {
    const strong = isDark() ? v / max > 0.55 : v / max > 0.5;
    annotations.push({
      x: names[j], y: names[i], showarrow: false,
      text: normalized ? v.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : String(v),
      font: { size: 12, color: strong ? (isDark() ? "#0b0b0b" : "#ffffff") : css("--text") },
    });
  }));
  plot("chart-cm", [{
    type: "heatmap", x: names, y: names, z, colorscale: scale, zmin: 0, zmax: max,
    xgap: 2, ygap: 2, showscale: false,
    customdata: cm.matrix,
    hovertemplate: "Asli <b>%{y}</b>, ditebak <b>%{x}</b><br>%{customdata} gambar" + (normalized ? " (%{z:.0%} dari kelas asli)" : "") + "<extra></extra>",
  }], baseLayout({
    annotations,
    xaxis: { ...baseLayout().xaxis, title: { text: "Prediksi model", font: { color: css("--text-2"), size: 12 } }, side: "bottom" },
    yaxis: { ...baseLayout().yaxis, title: { text: "Label asli", font: { color: css("--text-2"), size: 12 } }, autorange: "reversed", showgrid: false },
    margin: { t: 12, r: 12, b: 56, l: 76 },
  }));
}

// ------------------------------------------------------------ metrik per kelas
function perClassChart() {
  const report = state.data.perClass.report;
  const classes = state.data.confusion.classes;
  const names = classes.map(catName);
  const f1 = classes.map((c) => report[c]["f1-score"]);
  const markerLine = css("--surface");
  plot("chart-perclass", [
    {
      type: "bar", name: "F1", showlegend: false, x: names, y: f1,
      marker: { color: classes.map(catColor) },
      text: f1.map((v) => v.toLocaleString("id-ID", { minimumFractionDigits: 2, maximumFractionDigits: 2 })),
      textposition: "inside", insidetextanchor: "start",
      textfont: { color: classes.map((c) => contrastText(catColor(c))), size: 12 },
      customdata: classes.map((c) => [report[c].precision, report[c].recall, report[c].support]),
      hovertemplate: "<b>%{x}</b><br>F1 %{y:.3f}<br>Precision %{customdata[0]:.3f}<br>Recall %{customdata[1]:.3f}<br>%{customdata[2]} gambar test<extra></extra>",
    },
    {
      type: "scatter", mode: "markers", name: "Precision", x: names, y: classes.map((c) => report[c].precision),
      marker: { symbol: "circle", size: 11, color: css("--text"), line: { color: markerLine, width: 2 } },
      hovertemplate: "<b>%{x}</b><br>Precision %{y:.3f}<extra></extra>",
    },
    {
      type: "scatter", mode: "markers", name: "Recall", x: names, y: classes.map((c) => report[c].recall),
      marker: { symbol: "diamond", size: 12, color: css("--muted"), line: { color: markerLine, width: 2 } },
      hovertemplate: "<b>%{x}</b><br>Recall %{y:.3f}<extra></extra>",
    },
  ], baseLayout({
    yaxis: { ...baseLayout().yaxis, range: [0, 1.05], title: { text: "Skor di test set", font: { color: css("--text-2"), size: 12 } } },
    shapes: [refLine(0.7, "Batas minimum F1 0,70")],
    legend: { ...baseLayout().legend, y: 1.14 },
    margin: { t: 36, r: 12, b: 40, l: 60 },
  }));
}

// ------------------------------------------------------------ uji ketahanan
function robustnessChart() {
  const rows = state.data.robustness.results;
  const conds = [...new Set(rows.map((r) => r.gangguan))];
  const traces = MODEL_KEYS.map((k) => {
    const byCond = Object.fromEntries(rows.filter((r) => r.key === k).map((r) => [r.gangguan, r]));
    return {
      type: "bar", name: `${k}. ${MODEL_NAMES[k]}`, x: conds, y: conds.map((c) => byCond[c].accuracy),
      marker: { color: modelColor(k) },
      customdata: conds.map((c) => byCond[c].turun_akurasi * 100),
      hovertemplate: `<b>${MODEL_NAMES[k]}</b><br>%{x}<br>Accuracy %{y:.3f}<br>Turun %{customdata:.1f} poin dari gambar asli<extra></extra>`,
    };
  });
  plot("chart-robust", traces, baseLayout({
    barmode: "group",
    yaxis: { ...baseLayout().yaxis, range: [0, 1.05], title: { text: "Accuracy di test set", font: { color: css("--text-2"), size: 12 } } },
    xaxis: { ...baseLayout().xaxis, tickangle: 0 },
    legend: { ...baseLayout().legend, y: 1.16 },
    margin: { t: 40, r: 12, b: 64, l: 60 },
  }));
}

function renderAll() {
  if (!state.data || typeof Plotly === "undefined") return;
  datasetChart();
  modelsChart();
  trainingCharts();
  confusionChart();
  perClassChart();
  robustnessChart();
}

function bindControls() {
  for (const btn of document.querySelectorAll("[data-model-toggle]")) {
    btn.addEventListener("click", () => {
      const k = btn.dataset.modelToggle;
      if (state.visibleModels.has(k) && state.visibleModels.size === 1) return; // minimal satu model tampil
      state.visibleModels.has(k) ? state.visibleModels.delete(k) : state.visibleModels.add(k);
      btn.setAttribute("aria-pressed", String(state.visibleModels.has(k)));
      trainingCharts();
    });
  }
  for (const btn of document.querySelectorAll("[data-cm-mode]")) {
    btn.addEventListener("click", () => {
      state.cmMode = btn.dataset.cmMode;
      for (const b of document.querySelectorAll("[data-cm-mode]")) b.setAttribute("aria-pressed", String(b === btn));
      confusionChart();
    });
  }
}

export function initCharts(data) {
  state.data = data;
  bindControls();
  renderAll();
  document.addEventListener("themechange", renderAll);
}
