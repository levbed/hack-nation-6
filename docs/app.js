const DATA_URL = "data/multicohort_summary.json";

const COHORT_LABELS = {
  mcphases: "mcPHASES multimodal",
  utah_cycle_length: "Utah history replication",
};

const METRICS = {
  mae: { label: "Mean absolute error", short: "MAE", unit: "days", better: "lower", digits: 2 },
  rmse: { label: "Root mean squared error", short: "RMSE", unit: "days", better: "lower", digits: 2 },
  median_absolute_error: { label: "Median absolute error", short: "Median AE", unit: "days", better: "lower", digits: 2 },
  within_3_days_pct: { label: "Predictions within 3 days", short: "Within 3 days", unit: "%", better: "higher", digits: 1 },
  within_7_days_pct: { label: "Predictions within 7 days", short: "Within 7 days", unit: "%", better: "higher", digits: 1 },
  mean_signed_error: { label: "Mean signed error", short: "Signed error", unit: "days", better: "closer to zero", digits: 2 },
};

const TRACK_LABELS = {
  history_only: "History only",
  history_plus_wearables: "History + wearables",
  history_plus_hormones: "History + hormones",
  history_plus_symptoms: "History + symptoms",
  history_plus_glucose_stress: "History + glucose/stress",
  full_multimodal: "Full multimodal",
  global_median: "Global median",
  previous_cycle: "Previous cycle",
};

const state = {
  model: "ridge",
  metric: "mae",
  view: "ranking",
  delta: false,
  showCI: true,
  includeBaselines: true,
  tracks: new Set(),
};

let benchmark;
let multicohort;

const svgNS = "http://www.w3.org/2000/svg";
const el = (id) => document.getElementById(id);

function svgElement(name, attributes = {}, text = "") {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text) node.textContent = text;
  return node;
}

function formatValue(value, metric = state.metric, signed = false) {
  const definition = METRICS[metric];
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(definition.digits)}${definition.unit === "%" ? "%" : ""}`;
}

function trainedScores() {
  return benchmark.scores.filter((row) => row.model !== "baseline");
}

function modelReference(row, metric) {
  const referenceModel = row.model === "baseline" ? "ridge" : row.model;
  return benchmark.scores.find((candidate) => candidate.model === referenceModel && candidate.track === "history_only")?.[metric] ?? 0;
}

function displayValue(row) {
  const value = Number(row[state.metric]);
  return state.delta ? value - modelReference(row, state.metric) : value;
}

function selectedRows() {
  const rows = benchmark.scores.filter(
    (row) => row.model === state.model && state.tracks.has(row.track)
  );
  if (state.includeBaselines && state.view === "ranking") {
    rows.push(...benchmark.scores.filter((row) => row.model === "baseline"));
  }
  const metric = METRICS[state.metric];
  return rows.sort((a, b) => {
    const av = displayValue(a);
    const bv = displayValue(b);
    if (metric.better === "higher") return bv - av;
    if (metric.better === "closer to zero") return Math.abs(av) - Math.abs(bv);
    return av - bv;
  });
}

function renderSummary() {
  const cohort = benchmark.cohort_flow;
  el("participants").textContent = cohort.participants;
  el("cycles").textContent = cohort.inferred_complete_cycles;
  el("examples").textContent = cohort.eligible_examples;
  el("protocol").textContent = `Protocol ${benchmark.protocol_version}`;
  const best = [...trainedScores()].sort((a, b) => a.mae - b.mae)[0];
  el("finding").textContent = `Lowest point MAE: ${best.model_label} with ${TRACK_LABELS[best.track].toLowerCase()} (${best.mae.toFixed(2)} days).`;
}

function selectCohort(cohortId) {
  const selected = multicohort.cohorts.find((cohort) => cohort.dataset.id === cohortId);
  if (!selected) return;
  benchmark = selected;
  state.tracks = new Set(Object.keys(benchmark.feature_counts_by_track));
  renderSummary();
  render();
}

function renderCohortControls() {
  const control = el("cohort-control");
  control.replaceChildren();
  multicohort.cohorts.forEach((cohort) => {
    const cohortId = cohort.dataset.id;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = COHORT_LABELS[cohortId] || cohortId;
    button.className = cohortId === benchmark.dataset.id ? "active" : "";
    button.setAttribute("aria-pressed", String(cohortId === benchmark.dataset.id));
    button.addEventListener("click", () => selectCohort(cohortId));
    control.appendChild(button);
  });
}

function renderModelControls() {
  const control = el("model-control");
  control.replaceChildren();
  benchmark.evaluation.models.forEach((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = model.label;
    button.className = model.id === state.model ? "active" : "";
    button.setAttribute("aria-pressed", String(model.id === state.model));
    button.addEventListener("click", () => {
      state.model = model.id;
      render();
    });
    control.appendChild(button);
  });
}

function renderTrackControls() {
  const control = el("track-control");
  control.replaceChildren();
  Object.keys(benchmark.feature_counts_by_track).forEach((track) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.tracks.has(track);
    input.addEventListener("change", () => {
      if (input.checked) state.tracks.add(track);
      else state.tracks.delete(track);
      renderChartAndTable();
    });
    label.append(input, document.createTextNode(TRACK_LABELS[track]));
    control.appendChild(label);
  });
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  if (domainMin === domainMax) return () => (rangeMin + rangeMax) / 2;
  return (value) => rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function axisTicks(minimum, maximum, count = 5) {
  const span = maximum - minimum || 1;
  const rough = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(Math.abs(rough)));
  const normalized = rough / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  const first = Math.ceil(minimum / step) * step;
  const values = [];
  for (let value = first; value <= maximum + step * 0.01; value += step) values.push(value);
  return values;
}

function renderRanking(svg) {
  const rows = selectedRows();
  const width = 1000;
  const left = 230;
  const right = 80;
  const top = 28;
  const rowHeight = 48;
  const bottom = 54;
  const height = Math.max(410, top + rows.length * rowHeight + bottom);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.minHeight = `${Math.min(620, height)}px`;
  svg.replaceChildren();

  const values = rows.map(displayValue);
  const ciAvailable = state.metric === "mae" && state.showCI && !state.delta;
  const lowerValues = ciAvailable ? rows.map((row) => Number(row.mae_ci_low)) : values;
  const upperValues = ciAvailable ? rows.map((row) => Number(row.mae_ci_high)) : values;
  let minimum = Math.min(...lowerValues, state.delta ? 0 : 0);
  let maximum = Math.max(...upperValues, state.delta ? 0 : 1);
  const padding = Math.max((maximum - minimum) * 0.1, 0.4);
  if (state.delta) { minimum -= padding; maximum += padding; }
  else maximum += padding;
  const x = scaleLinear(minimum, maximum, left, width - right);

  axisTicks(minimum, maximum).forEach((tick) => {
    svg.appendChild(svgElement("line", { x1: x(tick), x2: x(tick), y1: top - 8, y2: height - bottom, class: tick === 0 ? "svg-zero" : "svg-grid" }));
    svg.appendChild(svgElement("text", { x: x(tick), y: height - 24, "text-anchor": "middle", class: "svg-axis" }, formatValue(tick)));
  });

  rows.forEach((row, index) => {
    const value = values[index];
    const y = top + index * rowHeight + 7;
    const zero = x(0);
    const valueX = x(value);
    const start = state.delta ? Math.min(zero, valueX) : x(0);
    const barWidth = Math.max(2, Math.abs(valueX - (state.delta ? zero : x(0))));
    const color = row.model === "baseline" ? "#d3a62f" : value < 0 && state.delta ? "#147d75" : row.model === "ridge" ? "#3568a8" : "#147d75";
    const bar = svgElement("rect", { x: start, y, width: barWidth, height: 25, rx: 2, fill: color, opacity: 0.9 });
    bar.appendChild(svgElement("title", {}, `${row.model_label}, ${TRACK_LABELS[row.track]}: ${formatValue(value)}`));
    svg.appendChild(bar);
    svg.appendChild(svgElement("text", { x: left - 12, y: y + 17, "text-anchor": "end", class: "svg-label" }, `${row.model_label} · ${TRACK_LABELS[row.track]}`));
    svg.appendChild(svgElement("text", { x: Math.min(width - 6, Math.max(left + 6, valueX + (value >= 0 ? 8 : -8))), y: y + 17, "text-anchor": value >= 0 ? "start" : "end", class: "svg-value" }, formatValue(value, state.metric, state.delta)));

    if (ciAvailable) {
      const low = x(Number(row.mae_ci_low));
      const high = x(Number(row.mae_ci_high));
      const centerY = y + 12.5;
      svg.appendChild(svgElement("line", { x1: low, x2: high, y1: centerY, y2: centerY, class: "svg-ci" }));
      svg.appendChild(svgElement("line", { x1: low, x2: low, y1: centerY - 5, y2: centerY + 5, class: "svg-ci" }));
      svg.appendChild(svgElement("line", { x1: high, x2: high, y1: centerY - 5, y2: centerY + 5, class: "svg-ci" }));
    }
  });
}

function heatColor(value, minimum, maximum, better) {
  const ratio = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum);
  const quality = better === "higher" ? 1 - ratio : better === "closer to zero" ? Math.abs(value) / Math.max(Math.abs(minimum), Math.abs(maximum)) : ratio;
  const bad = [198, 93, 72];
  const good = [52, 125, 116];
  const mix = (index) => Math.round(good[index] + (bad[index] - good[index]) * quality);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

function renderHeatmap(svg) {
  const models = benchmark.evaluation.models;
  const tracks = Object.keys(benchmark.feature_counts_by_track).filter((track) => state.tracks.has(track));
  const width = 1000;
  const left = 195;
  const top = 92;
  const right = 24;
  const bottom = 32;
  const cellWidth = (width - left - right) / Math.max(1, tracks.length);
  const cellHeight = 90;
  const height = top + models.length * cellHeight + bottom;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.minHeight = `${height}px`;
  svg.replaceChildren();

  const cells = [];
  models.forEach((model, rowIndex) => {
    tracks.forEach((track, columnIndex) => {
      const score = benchmark.scores.find((item) => item.model === model.id && item.track === track);
      if (score) cells.push({ model, track, rowIndex, columnIndex, score, value: displayValue(score) });
    });
  });
  const values = cells.map((cell) => cell.value);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);

  tracks.forEach((track, index) => {
    const text = svgElement("text", { x: left + index * cellWidth + cellWidth / 2, y: top - 18, "text-anchor": "middle", class: "svg-label" }, TRACK_LABELS[track]);
    text.setAttribute("transform", `rotate(-24 ${left + index * cellWidth + cellWidth / 2} ${top - 18})`);
    svg.appendChild(text);
  });
  models.forEach((model, index) => {
    svg.appendChild(svgElement("text", { x: left - 14, y: top + index * cellHeight + cellHeight / 2 + 5, "text-anchor": "end", class: "svg-label" }, model.label));
  });

  cells.forEach((cell) => {
    const x = left + cell.columnIndex * cellWidth;
    const y = top + cell.rowIndex * cellHeight;
    const rect = svgElement("rect", { x, y, width: cellWidth - 2, height: cellHeight - 2, rx: 2, fill: heatColor(cell.value, minimum, maximum, METRICS[state.metric].better) });
    rect.appendChild(svgElement("title", {}, `${cell.model.label}, ${TRACK_LABELS[cell.track]}: ${formatValue(cell.value)}`));
    svg.appendChild(rect);
    svg.appendChild(svgElement("text", { x: x + cellWidth / 2, y: y + cellHeight / 2 + 5, "text-anchor": "middle", fill: "#fff", "font-size": "14", "font-weight": "700" }, formatValue(cell.value, state.metric, state.delta)));
  });
}

function renderTable() {
  const body = el("result-table");
  body.replaceChildren();
  selectedRows().forEach((row) => {
    const tr = document.createElement("tr");
    const values = [
      row.model_label,
      TRACK_LABELS[row.track],
      formatValue(Number(row[state.metric])),
      `${row.mae_ci_low.toFixed(2)}–${row.mae_ci_high.toFixed(2)}`,
      formatValue(Number(row.delta_mae_vs_history), "mae", true),
      formatValue(Number(row.within_7_days_pct), "within_7_days_pct"),
    ];
    values.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index >= 2) td.className = "numeric";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function renderChartAndTable() {
  const metric = METRICS[state.metric];
  el("chart-title").textContent = state.delta ? `${metric.label}: difference from history` : metric.label;
  el("chart-direction").textContent = `${metric.better[0].toUpperCase()}${metric.better.slice(1)} is better`;
  el("metric-column").textContent = metric.short;
  el("ci-control").disabled = state.metric !== "mae" || state.delta || state.view === "heatmap";
  el("baseline-control").disabled = state.view === "heatmap";
  el("chart-note").textContent = state.metric === "mae" && state.showCI && !state.delta && state.view === "ranking"
    ? "Error bars are 95% participant-clustered bootstrap intervals."
    : state.delta
      ? `Differences use the selected model's history-only result as reference; ${metric.better === "higher" ? "positive" : metric.better === "lower" ? "negative" : "values near zero"} differences are better.`
      : "All values are aggregate out-of-fold estimates.";
  const chart = el("chart");
  if (state.view === "ranking") renderRanking(chart);
  else renderHeatmap(chart);
  renderTable();
}

function render() {
  renderCohortControls();
  renderModelControls();
  renderTrackControls();
  renderChartAndTable();
}

function bindControls() {
  el("metric-control").addEventListener("change", (event) => {
    state.metric = event.target.value;
    renderChartAndTable();
  });
  el("delta-control").addEventListener("change", (event) => {
    state.delta = event.target.checked;
    renderChartAndTable();
  });
  el("ci-control").addEventListener("change", (event) => {
    state.showCI = event.target.checked;
    renderChartAndTable();
  });
  el("baseline-control").addEventListener("change", (event) => {
    state.includeBaselines = event.target.checked;
    renderChartAndTable();
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      document.querySelectorAll(".tab").forEach((candidate) => {
        const selected = candidate === button;
        candidate.classList.toggle("active", selected);
        candidate.setAttribute("aria-selected", String(selected));
      });
      renderChartAndTable();
    });
  });
}

async function initialize() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    multicohort = await response.json();
    if (!Array.isArray(multicohort.cohorts) || !multicohort.cohorts.length) {
      throw new Error("No cohort summaries found");
    }
    benchmark = multicohort.cohorts[0];
    state.tracks = new Set(Object.keys(benchmark.feature_counts_by_track));
    renderSummary();
    bindControls();
    render();
  } catch (error) {
    el("finding").textContent = "Aggregate results could not be loaded.";
    el("chart-note").textContent = String(error);
  }
}

initialize();
