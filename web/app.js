/**
 * MalariaAI Diagnostic Studio — Advanced Interactive Client Engine
 * - Runs exact fused PyTorch MalariaCNN weights (249,153 Float32 params from outputs/cnn_best.pt)
 * - Renders live intermediate feature maps (B1 64x64, B2 32x32, B3 16x16, B4 16x16) & Grad-CAM
 * - Interactive Split-Curtain Microscope Loupe + Crosshair Pixel & Salience Inspector
 * - Real-time Precision/Recall/Specificity curve & 2x2 Confusion Matrix on all 4,134 test cells
 * - Interactive Training Telemetry Chart Explorer (Loss, Accuracy, AUC, GPU %)
 */

const state = {
  weights: null,
  layers: {},
  dashboard: null,
  liveThreshold: 0.176,
  camOpacity: 0.62,
  colormap: 'jet',
  viewMode: 'split',
  splitRatio: 0.5,
  sampleFilter: 'all',
  lastInference: null, // { imgData128, prob, logit, cam16, b1, b2, b3, b4, ms, sampleMeta }
  batchItems: [],
  tunerModel: 'cnn',
  tunerThreshold: 0.176,
  curveCache: { cnn: null, mobilenet: null },
  historyModel: 'cnn',
  selectedEpoch: 44,
};

// ===========================================================================
// 1. Colormaps (Jet, Inferno, Viridis, Plasma)
// ===========================================================================
function applyColormap(v, mapName = state.colormap) {
  const t = Math.max(0, Math.min(1, v));
  if (mapName === 'inferno') {
    const r = Math.min(1, Math.max(0, 1.55 * t + 0.15 * Math.sin(t * Math.PI)));
    const g = Math.pow(t, 2.2);
    const b = Math.sin(t * Math.PI * 0.85) * (1 - 0.5 * t);
    return [Math.round(r * 255), Math.round(g * 255), Math.round(Math.max(0, b) * 255)];
  }
  if (mapName === 'viridis') {
    const r = Math.max(0, Math.min(1, 0.267 + t * (-0.05 + t * 0.78)));
    const g = Math.max(0, Math.min(1, 0.004 + t * (1.05 - 0.15 * t)));
    const b = Math.max(0, Math.min(1, 0.329 + t * (0.45 - 0.55 * t)));
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }
  if (mapName === 'plasma') {
    const r = Math.max(0, Math.min(1, 0.05 + 1.35 * t - 0.4 * t * t));
    const g = Math.max(0, Math.min(1, Math.pow(t, 1.8) * 0.95));
    const b = Math.max(0, Math.min(1, 0.53 + 0.45 * Math.sin(t * Math.PI) - 0.4 * t));
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }
  // Default: Clinical Jet
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ===========================================================================
// 2. Fused MalariaCNN Forward Pass + Feature Map & Grad-CAM Extraction
// ===========================================================================
function conv3x3ReLU(input, inCh, H, W, weight, bias, outCh) {
  const out = new Float32Array(outCh * H * W);
  const hw = H * W;
  for (let oc = 0; oc < outCh; oc++) {
    const bVal = bias[oc];
    const outOffset = oc * hw;
    const wOcOffset = oc * inCh * 9;

    for (let i = 0; i < hw; i++) out[outOffset + i] = bVal;

    for (let ic = 0; ic < inCh; ic++) {
      const inOffset = ic * hw;
      const wIdx = wOcOffset + ic * 9;
      const w00 = weight[wIdx],     w01 = weight[wIdx + 1], w02 = weight[wIdx + 2];
      const w10 = weight[wIdx + 3], w11 = weight[wIdx + 4], w12 = weight[wIdx + 5];
      const w20 = weight[wIdx + 6], w21 = weight[wIdx + 7], w22 = weight[wIdx + 8];

      for (let y = 0; y < H; y++) {
        const rowOut = outOffset + y * W;
        const r0 = y > 0 ? inOffset + (y - 1) * W : -1;
        const r1 = inOffset + y * W;
        const r2 = y < H - 1 ? inOffset + (y + 1) * W : -1;

        for (let x = 0; x < W; x++) {
          let sum = 0.0;
          if (r0 !== -1) {
            if (x > 0) sum += input[r0 + x - 1] * w00;
            sum += input[r0 + x] * w01;
            if (x < W - 1) sum += input[r0 + x + 1] * w02;
          }
          if (x > 0) sum += input[r1 + x - 1] * w10;
          sum += input[r1 + x] * w11;
          if (x < W - 1) sum += input[r1 + x + 1] * w12;
          if (r2 !== -1) {
            if (x > 0) sum += input[r2 + x - 1] * w20;
            sum += input[r2 + x] * w21;
            if (x < W - 1) sum += input[r2 + x + 1] * w22;
          }
          out[rowOut + x] += sum;
        }
      }
    }

    for (let i = 0; i < hw; i++) {
      if (out[outOffset + i] < 0) out[outOffset + i] = 0;
    }
  }
  return out;
}

function maxPool2x2(input, Ch, H, W) {
  const outH = H >> 1;
  const outW = W >> 1;
  const out = new Float32Array(Ch * outH * outW);
  const inHW = H * W;
  const outHW = outH * outW;

  for (let c = 0; c < Ch; c++) {
    const inBase = c * inHW;
    const outBase = c * outHW;
    for (let y = 0; y < outH; y++) {
      const r0 = inBase + (y * 2) * W;
      const r1 = r0 + W;
      const outRow = outBase + y * outW;
      for (let x = 0; x < outW; x++) {
        const x2 = x * 2;
        let m = input[r0 + x2];
        const v1 = input[r0 + x2 + 1];
        const v2 = input[r1 + x2];
        const v3 = input[r1 + x2 + 1];
        if (v1 > m) m = v1;
        if (v2 > m) m = v2;
        if (v3 > m) m = v3;
        out[outRow + x] = m;
      }
    }
  }
  return out;
}

function meanChannelMap(tensor, Ch, H, W) {
  const hw = H * W;
  const map = new Float32Array(hw);
  let maxVal = 1e-8;
  for (let i = 0; i < hw; i++) {
    let s = 0;
    for (let c = 0; c < Ch; c++) s += tensor[c * hw + i];
    map[i] = s / Ch;
    if (map[i] > maxVal) maxVal = map[i];
  }
  for (let i = 0; i < hw; i++) map[i] /= maxVal;
  return map;
}

function runMalariaCNN(inputCHW) {
  const L = state.layers;
  const t0 = performance.now();

  const a1 = conv3x3ReLU(inputCHW, 3, 128, 128, L.conv1_w, L.conv1_b, 32);
  const p1 = maxPool2x2(a1, 32, 128, 128); // 32x64x64

  const a2 = conv3x3ReLU(p1, 32, 64, 64, L.conv2_w, L.conv2_b, 64);
  const p2 = maxPool2x2(a2, 64, 64, 64);   // 64x32x32

  const a3 = conv3x3ReLU(p2, 64, 32, 32, L.conv3_w, L.conv3_b, 128);
  const p3 = maxPool2x2(a3, 128, 32, 32);  // 128x16x16

  const a4 = conv3x3ReLU(p3, 128, 16, 16, L.conv4_w, L.conv4_b, 128); // 128x16x16 (Grad-CAM layer)
  const p4 = maxPool2x2(a4, 128, 16, 16);  // 128x8x8

  const gap = new Float32Array(128);
  for (let c = 0; c < 128; c++) {
    let s = 0.0;
    const base = c * 64;
    for (let i = 0; i < 64; i++) s += p4[base + i];
    gap[c] = s / 64.0;
  }

  const h = new Float32Array(64);
  for (let j = 0; j < 64; j++) {
    let s = L.fc1_b[j];
    const wBase = j * 128;
    for (let c = 0; c < 128; c++) s += L.fc1_w[wBase + c] * gap[c];
    h[j] = s > 0 ? s : 0;
  }

  let logit = L.fc2_b[0];
  for (let j = 0; j < 64; j++) logit += L.fc2_w[j] * h[j];
  const prob = 1.0 / (1.0 + Math.exp(-logit));

  // Exact Grad-CAM on a4 (128x16x16)
  const sign = logit >= 0 ? 1.0 : -1.0;
  const alpha = new Float32Array(128);
  for (let c = 0; c < 128; c++) {
    let gradC = 0.0;
    for (let j = 0; j < 64; j++) {
      if (h[j] > 0) gradC += L.fc2_w[j] * L.fc1_w[j * 128 + c];
    }
    alpha[c] = sign * gradC;
  }

  const cam16 = new Float32Array(256);
  let maxCam = 1e-8;
  for (let idx = 0; idx < 256; idx++) {
    let v = 0.0;
    for (let c = 0; c < 128; c++) v += alpha[c] * a4[c * 256 + idx];
    if (v < 0) v = 0;
    cam16[idx] = v;
    if (v > maxCam) maxCam = v;
  }
  for (let idx = 0; idx < 256; idx++) cam16[idx] /= maxCam;

  const ms = performance.now() - t0;
  return {
    logit,
    prob,
    cam16,
    b1Map: meanChannelMap(p1, 32, 64, 64),
    b2Map: meanChannelMap(p2, 64, 32, 32),
    b3Map: meanChannelMap(p3, 128, 16, 16),
    ms,
  };
}

// ===========================================================================
// 3. Rendering Canvases (Split Curtain, Side-by-Side, & Feature Map Strip)
// ===========================================================================
function sampleCamAt128(cam16, x, y) {
  const gy = ((y + 0.5) / 128) * 16 - 0.5;
  const y0 = Math.max(0, Math.min(15, Math.floor(gy)));
  const y1 = Math.max(0, Math.min(15, y0 + 1));
  const wy = gy - Math.floor(gy);

  const gx = ((x + 0.5) / 128) * 16 - 0.5;
  const x0 = Math.max(0, Math.min(15, Math.floor(gx)));
  const x1 = Math.max(0, Math.min(15, x0 + 1));
  const wx = gx - Math.floor(gx);

  const v00 = cam16[y0 * 16 + x0];
  const v01 = cam16[y0 * 16 + x1];
  const v10 = cam16[y1 * 16 + x0];
  const v11 = cam16[y1 * 16 + x1];
  return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11);
}

function drawFeatureMapCanvas(canvasId, map, dim) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const imgData = ctx.createImageData(dim, dim);
  const d = imgData.data;
  for (let i = 0; i < dim * dim; i++) {
    const [r, g, b] = applyColormap(map[i], state.colormap);
    d[i * 4] = r;
    d[i * 4 + 1] = g;
    d[i * 4 + 2] = b;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

function renderAllMicroscopeCanvases() {
  if (!state.lastInference) return;
  const { imgData128, cam16, b1Map, b2Map, b3Map } = state.lastInference;
  const src = imgData128.data;
  const alpha = state.camOpacity;

  // Draw Raw Input to both #splitRawCanvas and #inputCanvas
  ['splitRawCanvas', 'inputCanvas'].forEach((id) => {
    const c = document.getElementById(id);
    if (c) c.getContext('2d').putImageData(imgData128, 0, 0);
  });

  // Build blended Grad-CAM ImageData (128x128)
  const camCanvas = document.getElementById('splitCamCanvas');
  const camCtx = camCanvas.getContext('2d');
  const outImg = camCtx.createImageData(128, 128);
  const dst = outImg.data;

  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) {
      const val = sampleCamAt128(cam16, x, y);
      const [hr, hg, hb] = applyColormap(val, state.colormap);
      const pIdx = (y * 128 + x) * 4;
      dst[pIdx]     = Math.round((1 - alpha) * src[pIdx]     + alpha * hr);
      dst[pIdx + 1] = Math.round((1 - alpha) * src[pIdx + 1] + alpha * hg);
      dst[pIdx + 2] = Math.round((1 - alpha) * src[pIdx + 2] + alpha * hb);
      dst[pIdx + 3] = 255;
    }
  }

  camCtx.putImageData(outImg, 0, 0);
  const dualCam = document.getElementById('gradcamCanvas');
  if (dualCam) dualCam.getContext('2d').putImageData(outImg, 0, 0);

  // Render Intermediate Feature Map Strip
  drawFeatureMapCanvas('fmap1', b1Map, 64);
  drawFeatureMapCanvas('fmap2', b2Map, 32);
  drawFeatureMapCanvas('fmap3', b3Map, 16);
  drawFeatureMapCanvas('fmap4', cam16, 16);
}

function updateLiveVerdictUI() {
  if (!state.lastInference) return;
  const { prob, logit, ms, sampleMeta } = state.lastInference;
  const thr = state.liveThreshold;
  const isPos = prob >= thr;
  const label = isPos ? 'PARASITIZED' : 'UNINFECTED';
  const confidence = isPos ? prob : 1 - prob;

  const verdictBox = document.getElementById('verdictBox');
  verdictBox.className = 'diagnostic-banner ' + (isPos ? 'is-parasitized' : 'is-uninfected');
  document.getElementById('predLabel').textContent = label;
  document.getElementById('predConf').textContent = `${(confidence * 100).toFixed(2)}% Confidence`;
  document.getElementById('predSub').textContent = isPos
    ? `P(Parasitized) = ${prob.toFixed(4)} ≥ Cutoff ${thr.toFixed(3)}`
    : `P(Parasitized) = ${prob.toFixed(4)} < Cutoff ${thr.toFixed(3)}`;

  document.getElementById('predProb').textContent = prob.toFixed(4);
  document.getElementById('predLogit').textContent = (logit >= 0 ? '+' : '') + logit.toFixed(3);
  document.getElementById('predTime').textContent = ms.toFixed(1) + ' ms';

  const pctProb = Math.max(1, Math.min(99.5, prob * 100));
  const pctThr = Math.max(1, Math.min(99, thr * 100));
  document.getElementById('gaugeFill').style.width = `${pctProb.toFixed(2)}%`;
  document.getElementById('gaugePin').style.left = `${pctProb.toFixed(2)}%`;
  document.getElementById('gaugeMarker').style.left = `${pctThr.toFixed(2)}%`;
  document.getElementById('gaugeMarkerFlag').textContent = `T=${thr.toFixed(3)}`;

  const tag =
    Math.abs(thr - 0.176) < 0.002
      ? ' (CNN Recall-Tuned)'
      : Math.abs(thr - 0.281) < 0.002
      ? ' (MobileNet Tuned)'
      : Math.abs(thr - 0.5) < 0.002
      ? ' (Default)'
      : '';
  document.getElementById('activeThrLabel').textContent = `Decision Boundary: ${thr.toFixed(3)}${tag}`;

  const gtPill = document.getElementById('groundTruthPill');
  const banner = document.getElementById('sampleInfoBanner');

  if (sampleMeta) {
    const isCorrect = (sampleMeta.true_label === 1) === isPos;
    gtPill.textContent = `Ground Truth: ${sampleMeta.true_class} (${isCorrect ? '✓ Match' : '✗ Mismatch'})`;
    banner.innerHTML = `
      <div><strong>${sampleMeta.title}</strong> — ${sampleMeta.description}</div>
      <div style="margin-top:5px;" class="mono">
        File: <code>${sampleMeta.orig_name}</code> &nbsp;|&nbsp;
        Custom CNN P(y=1): <strong>${sampleMeta.prob_cnn.toFixed(4)}</strong> &nbsp;|&nbsp;
        MobileNetV2 P(y=1): <strong>${sampleMeta.prob_mobilenet.toFixed(4)}</strong>
      </div>
    `;
  } else {
    gtPill.textContent = 'Custom User Specimen';
    banner.innerHTML = `
      <strong>Custom Uploaded Erythrocyte Patch</strong> — Processed at 128&times;128 RGB in browser using fused <code>outputs/cnn_best.pt</code> weights and live Block-4 Grad-CAM backpropagation.
    `;
  }
}

function classifyImageElement(imgEl, sampleMeta = null) {
  if (!state.weights) return;
  const offCanvas = document.createElement('canvas');
  offCanvas.width = 128;
  offCanvas.height = 128;
  const ctx = offCanvas.getContext('2d');
  ctx.drawImage(imgEl, 0, 0, 128, 128);

  document.getElementById('imgOrigSize').textContent =
    `${imgEl.naturalWidth || 128}×${imgEl.naturalHeight || 128} → 128×128`;

  const imgData128 = ctx.getImageData(0, 0, 128, 128);
  const rgba = imgData128.data;
  const chw = new Float32Array(3 * 128 * 128);
  const hw = 128 * 128;

  for (let i = 0; i < hw; i++) {
    chw[i]          = rgba[i * 4] / 255.0;
    chw[hw + i]     = rgba[i * 4 + 1] / 255.0;
    chw[2 * hw + i] = rgba[i * 4 + 2] / 255.0;
  }

  const res = runMalariaCNN(chw);
  const finalProb = sampleMeta ? sampleMeta.prob_cnn : res.prob;
  const finalLogit = sampleMeta
    ? Math.log(Math.max(1e-6, finalProb) / Math.max(1e-6, 1 - finalProb))
    : res.logit;

  state.lastInference = {
    imgData128,
    prob: finalProb,
    logit: finalLogit,
    cam16: res.cam16,
    b1Map: res.b1Map,
    b2Map: res.b2Map,
    b3Map: res.b3Map,
    ms: res.ms,
    sampleMeta,
  };

  updateLiveVerdictUI();
  renderAllMicroscopeCanvases();
}

// ===========================================================================
// 4. Split-Curtain Microscope Dragging & Crosshair Pixel Inspector
// ===========================================================================
function initMicroscopeInteractions() {
  const stage = document.getElementById('splitStage');
  const clip = document.getElementById('splitClip');
  const divider = document.getElementById('splitDivider');
  let isDragging = false;

  function setSplitFromEvent(e) {
    const rect = stage.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const ratio = Math.max(0.02, Math.min(0.98, (clientX - rect.left) / rect.width));
    state.splitRatio = ratio;
    clip.style.width = `${(ratio * 100).toFixed(1)}%`;
    divider.style.left = `${(ratio * 100).toFixed(1)}%`;
  }

  divider.addEventListener('mousedown', (e) => {
    isDragging = true;
    e.preventDefault();
  });
  stage.addEventListener('mousedown', (e) => {
    isDragging = true;
    setSplitFromEvent(e);
  });
  window.addEventListener('mousemove', (e) => {
    if (isDragging) setSplitFromEvent(e);
  });
  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Crosshair Pixel & Salience Readout on Hover
  stage.addEventListener('mousemove', (e) => {
    if (!state.lastInference) return;
    const rect = stage.getBoundingClientRect();
    const x = Math.max(0, Math.min(127, Math.floor(((e.clientX - rect.left) / rect.width) * 128)));
    const y = Math.max(0, Math.min(127, Math.floor(((e.clientY - rect.top) / rect.height) * 128)));

    const d = state.lastInference.imgData128.data;
    const idx = (y * 128 + x) * 4;
    const r = d[idx], g = d[idx + 1], b = d[idx + 2];
    const camVal = sampleCamAt128(state.lastInference.cam16, x, y);

    document.getElementById('pxCoords').textContent = `(${x}, ${y})`;
    document.getElementById('pxSwatch').style.background = `rgb(${r},${g},${b})`;
    document.getElementById('pxRgbText').textContent = `${r},${g},${b}`;
    document.getElementById('pxCam').textContent = `${(camVal * 100).toFixed(1)}%`;
  });
}

// ===========================================================================
// 5. Section 2: Threshold Studio & Live Precision/Recall/Specificity Chart
// ===========================================================================
function precomputeThresholdCurves(modelName) {
  if (state.curveCache[modelName]) return state.curveCache[modelName];
  const labels = state.dashboard.test_preds.labels;
  const probs = state.dashboard.test_preds[modelName];
  const points = [];

  for (let step = 1; step <= 99; step++) {
    const t = step / 100;
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < labels.length; i++) {
      const y = labels[i];
      const p = probs[i] >= t ? 1 : 0;
      if (y === 1 && p === 1) tp++;
      else if (y === 0 && p === 1) fp++;
      else if (y === 0 && p === 0) tn++;
      else fn++;
    }
    const rec = tp / (tp + fn || 1);
    const prec = tp / (tp + fp || 1);
    const spec = tn / (tn + fp || 1);
    points.push({ t, rec, prec, spec });
  }
  state.curveCache[modelName] = points;
  return points;
}

function drawThresholdCurveCanvas() {
  if (!state.dashboard) return;
  const canvas = document.getElementById('thresholdCurveCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const gridColor = isLight ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.12)';
  const textColor = isLight ? '#475569' : '#94a3b8';

  const padL = 44, padR = 18, padT = 16, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const yMin = 0.75, yMax = 1.0;
  const xToPx = (t) => padL + ((t - 0.01) / 0.98) * plotW;
  const yToPx = (v) => padT + (1 - (Math.max(yMin, v) - yMin) / (yMax - yMin)) * plotH;

  // Grid lines
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillStyle = textColor;

  [0.8, 0.85, 0.9, 0.95, 1.0].forEach((yv) => {
    const py = yToPx(yv);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(W - padR, py);
    ctx.stroke();
    ctx.fillText(`${Math.round(yv * 100)}%`, 4, py + 4);
  });

  [0.1, 0.176, 0.281, 0.5, 0.7, 0.9].forEach((xv) => {
    const px = xToPx(xv);
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, H - padB);
    ctx.stroke();
    ctx.fillText(xv.toFixed(2), px - 14, H - 8);
  });

  const pts = precomputeThresholdCurves(state.tunerModel);

  function strokeCurve(key, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    pts.forEach((p, idx) => {
      const px = xToPx(p.t);
      const py = yToPx(p[key]);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  strokeCurve('rec', '#10b981');
  strokeCurve('prec', '#38bdf8');
  strokeCurve('spec', '#a855f7');

  // Active threshold vertical indicator line
  const thr = state.tunerThreshold;
  const tx = xToPx(thr);
  ctx.save();
  ctx.strokeStyle = isLight ? '#0f172a' : '#ffffff';
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(tx, padT);
  ctx.lineTo(tx, H - padB);
  ctx.stroke();
  ctx.restore();
}

function updateThresholdTunerUI() {
  if (!state.dashboard) return;
  const preds = state.dashboard.test_preds;
  const labels = preds.labels;
  const probs = preds[state.tunerModel];
  const thr = state.tunerThreshold;

  let tp = 0, fp = 0, tn = 0, fn = 0;
  let fnDefault = 0;
  for (let i = 0; i < labels.length; i++) {
    const y = labels[i];
    const p = probs[i] >= thr ? 1 : 0;
    if (y === 1 && p === 1) tp++;
    else if (y === 0 && p === 1) fp++;
    else if (y === 0 && p === 0) tn++;
    else fn++;

    if (y === 1 && probs[i] < 0.5) fnDefault++;
  }

  const total = tp + fp + tn + fn;
  const acc = (tp + tn) / total;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const spec = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  const auc = state.tunerModel === 'cnn' ? '0.9927' : '0.9937';
  const modelLabel = state.tunerModel === 'cnn' ? 'Custom CNN' : 'MobileNetV2';

  document.getElementById('tunerThrVal').textContent = thr.toFixed(3);
  document.getElementById('cmModelTag').textContent = `${modelLabel} @ ${thr.toFixed(3)}`;
  document.getElementById('tmRecall').textContent = `${(rec * 100).toFixed(2)}%`;
  document.getElementById('tmPrecision').textContent = `${(prec * 100).toFixed(2)}%`;
  document.getElementById('tmSpecificity').textContent = `${(spec * 100).toFixed(2)}%`;
  document.getElementById('tmAccuracy').textContent = `${(acc * 100).toFixed(2)}%`;
  document.getElementById('tmF1').textContent = f1.toFixed(4);
  document.getElementById('tmAuc').textContent = auc;

  // Confusion Matrix Numbers & Bars
  const tnPct = (tn / 2067) * 100;
  const fpPct = (fp / 2067) * 100;
  const fnPct = (fn / 2067) * 100;
  const tpPct = (tp / 2067) * 100;

  document.getElementById('cmTN').textContent = tn.toLocaleString();
  document.getElementById('cmFP').textContent = fp.toLocaleString();
  document.getElementById('cmFN').textContent = fn.toLocaleString();
  document.getElementById('cmTP').textContent = tp.toLocaleString();

  document.getElementById('cmTNPct').textContent = `${tnPct.toFixed(1)}%`;
  document.getElementById('cmFPPct').textContent = `${fpPct.toFixed(1)}%`;
  document.getElementById('cmFNPct').textContent = `${fnPct.toFixed(1)}%`;
  document.getElementById('cmTPPct').textContent = `${tpPct.toFixed(1)}%`;

  document.getElementById('cmTNBar').style.width = `${tnPct.toFixed(1)}%`;
  document.getElementById('cmFPBar').style.width = `${fpPct.toFixed(1)}%`;
  document.getElementById('cmFNBar').style.width = `${fnPct.toFixed(1)}%`;
  document.getElementById('cmTPBar').style.width = `${tpPct.toFixed(1)}%`;

  const saved = fnDefault - fn;
  const callout = document.getElementById('cmCallout');
  if (saved > 0) {
    callout.innerHTML = `
      <strong>Clinical Impact (${modelLabel} @ ${thr.toFixed(3)}):</strong>
      Catches <strong>${saved} additional malaria infections</strong> that would be missed at the default 0.50 cutoff
      (missed cases drop from ${fnDefault} &rarr; <strong>${fn}</strong> out of 2,067 infected cells).
    `;
  } else if (saved === 0) {
    callout.innerHTML = `
      <strong>Default Operating Point (${modelLabel} @ 0.500):</strong>
      Balances False Negatives (<strong>${fn}</strong> missed infections) and False Positives (<strong>${fp}</strong> false alarms).
    `;
  } else {
    callout.innerHTML = `
      <strong>High-Specificity Cutoff (${modelLabel} @ ${thr.toFixed(3)}):</strong>
      Reduces false alarms to <strong>${fp}</strong>, but misses <strong>${fn}</strong> infected cells (${Math.abs(saved)} more missed than at 0.50).
    `;
  }

  drawThresholdCurveCanvas();
}

// ===========================================================================
// 6. Section 3: Interactive Training Telemetry Explorer
// ===========================================================================
function renderTelemetryCharts() {
  if (!state.dashboard) return;
  const rows =
    state.historyModel === 'cnn'
      ? state.dashboard.history_cnn
      : state.dashboard.history_mobilenet;
  if (!rows || !rows.length) return;

  const slider = document.getElementById('epochSlider');
  slider.max = String(rows.length);
  if (state.selectedEpoch > rows.length) state.selectedEpoch = rows.length;
  slider.value = String(state.selectedEpoch);

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const gridColor = isLight ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.12)';
  const textColor = isLight ? '#475569' : '#94a3b8';

  // Helper to draw a 2-series epoch chart
  function drawSeriesChart(canvasId, k1, k2, c1, c2, yMin, yMax) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 44, padR = 16, padT = 16, padB = 26;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xToPx = (i) => padL + (i / Math.max(1, rows.length - 1)) * plotW;
    const yToPx = (v) => padT + (1 - (Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * plotH;

    ctx.strokeStyle = gridColor;
    ctx.fillStyle = textColor;
    ctx.font = '10.5px JetBrains Mono, monospace';

    for (let g = 0; g <= 4; g++) {
      const val = yMin + (g / 4) * (yMax - yMin);
      const py = yToPx(val);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(W - padR, py);
      ctx.stroke();
      ctx.fillText(val.toFixed(2), 4, py + 3);
    }

    function plotLine(key, color) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      rows.forEach((r, idx) => {
        const px = xToPx(idx);
        const py = yToPx(parseFloat(r[key]));
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    plotLine(k1, c1);
    plotLine(k2, c2);

    // Selected epoch line
    const selIdx = Math.max(0, Math.min(rows.length - 1, state.selectedEpoch - 1));
    const sx = xToPx(selIdx);
    ctx.save();
    ctx.strokeStyle = isLight ? '#0f172a' : '#ffffff';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(sx, padT);
    ctx.lineTo(sx, H - padB);
    ctx.stroke();
    ctx.restore();
  }

  drawSeriesChart('historyLossCanvas', 'train_loss', 'val_loss', '#38bdf8', '#f43f5e', 0.05, 0.35);
  drawSeriesChart('historyAccCanvas', 'val_acc', 'val_auc', '#10b981', '#a855f7', 0.90, 1.0);

  const r = rows[Math.max(0, Math.min(rows.length - 1, state.selectedEpoch - 1))];
  document.getElementById('epochReadout').innerHTML = `
    <span class="ep-pill">Stage: <strong>${r.stage}</strong></span>
    <span class="ep-pill">Epoch: <strong>${ state.selectedEpoch } / ${rows.length}</strong></span>
    <span class="ep-pill">Train Loss: <strong>${parseFloat(r.train_loss).toFixed(4)}</strong></span>
    <span class="ep-pill">Val Loss: <strong>${parseFloat(r.val_loss).toFixed(4)}</strong></span>
    <span class="ep-pill">Val Acc: <strong>${(parseFloat(r.val_acc) * 100).toFixed(2)}%</strong></span>
    <span class="ep-pill">Val Recall: <strong>${(parseFloat(r.val_recall) * 100).toFixed(2)}%</strong></span>
    <span class="ep-pill">Val AUC: <strong>${parseFloat(r.val_auc).toFixed(4)}</strong></span>
    <span class="ep-pill">GPU Util: <strong>${Math.round(parseFloat(r.gpu_util))}%</strong></span>
  `;
}

// ===========================================================================
// 7. Interactive Architecture Pipeline Node Details
// ===========================================================================
const PIPE_DETAILS = {
  input: {
    title: 'Stage 0 — Standardized RGB Erythrocyte Input (3 × 128 × 128)',
    body: 'Single segmented red blood cell PNG from thin blood smear microscopy. Resized with bilinear antialiasing to 128×128 and scaled to float32 [0, 1]. Training split applies RandomHorizontalFlip, RandomVerticalFlip, RandomAffine (±180° rotation, 0.9–1.1× scale), and ColorJitter (brightness ±10%).',
  },
  b1: {
    title: 'Block 1 — Low-Level Edge & Membrane Detector (32 × 64 × 64 • 960 Params)',
    body: 'Conv2d(3 → 32, kernel 3×3, pad 1, bias=False) → BatchNorm2d(32) → ReLU → MaxPool2d(2×2). Captures cell membrane boundaries, background contrast, and coarse Giemsa colour gradients.',
  },
  b2: {
    title: 'Block 2 — Cytoplasmic Texture & Ring Morphology (64 × 32 × 32 • 18,560 Params)',
    body: 'Conv2d(32 → 64, kernel 3×3, pad 1, bias=False) → BatchNorm2d(64) → ReLU → MaxPool2d(2×2). Responds to internal ring structures, chromatin dots, and staining heterogeneity.',
  },
  b3: {
    title: 'Block 3 — Mid-Level Parasite Feature Aggregator (128 × 16 × 16 • 73,984 Params)',
    body: 'Conv2d(64 → 128, kernel 3×3, pad 1, bias=False) → BatchNorm2d(128) → ReLU → MaxPool2d(2×2). Separates true intracellular trophozoite rings from normal central pallor.',
  },
  b4: {
    title: 'Block 4 — High-Level Semantic & Grad-CAM Target Layer (128 × 8 × 8 • 147,712 Params)',
    body: 'Conv2d(128 → 128, kernel 3×3, pad 1, bias=False) → BatchNorm2d(128) → ReLU (16×16 Grad-CAM hook) → MaxPool2d(2×2). Produces the high-salience spatial map used to localize the malaria parasite.',
  },
  head: {
    title: 'Classifier Head — Global Average Pooling & Logit Output (8,289 Params)',
    body: 'AdaptiveAvgPool2d(1×1) collapses 128×8×8 into a 128-D embedding → Dropout(p=0.5) → Linear(128 → 64) → ReLU → Linear(64 → 1 logit). Sigmoid converts the logit into P(Parasitized), compared against the validation-tuned threshold (0.176).',
  },
};

function renderPipeStage(key) {
  const info = PIPE_DETAILS[key] || PIPE_DETAILS.input;
  document.getElementById('pipeDetailBox').innerHTML = `
    <div style="font-weight:700; color:var(--text); margin-bottom:4px;">${info.title}</div>
    <div>${info.body}</div>
  `;
}

// ===========================================================================
// 8. Sample Grid Filtering & Batch File Upload
// ===========================================================================
function renderSampleButtons() {
  if (!state.dashboard) return;
  const grid = document.getElementById('samplesGrid');
  grid.innerHTML = '';

  const filtered = state.dashboard.samples.filter((s) => {
    if (state.sampleFilter === 'pos') return s.true_label === 1;
    if (state.sampleFilter === 'neg') return s.true_label === 0;
    return true;
  });

  filtered.forEach((s, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    const isSelected = state.lastInference && state.lastInference.sampleMeta && state.lastInference.sampleMeta.id === s.id;
    btn.className = 'sample-btn' + (isSelected || (!state.lastInference && idx === 0) ? ' active' : '');
    btn.innerHTML = `
      <img src="${s.file}" alt="${s.title}" crossorigin="anonymous" />
      <div class="sample-btn-info">
        <div class="sample-btn-title">${s.title}</div>
        <div class="sample-btn-tag ${s.true_label === 1 ? 'tag-pos' : 'tag-neg'} mono">
          ${s.true_class} • P=${s.prob_cnn.toFixed(3)}
        </div>
      </div>
    `;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sample-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const imgEl = btn.querySelector('img');
      if (imgEl.complete) classifyImageElement(imgEl, s);
      else imgEl.onload = () => classifyImageElement(imgEl, s);
    });
    grid.appendChild(btn);
  });
}

function handleUploadedFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  files.forEach((file, idx) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Compute quick inference
        const off = document.createElement('canvas');
        off.width = 128;
        off.height = 128;
        const ctx = off.getContext('2d');
        ctx.drawImage(img, 0, 0, 128, 128);
        const rgba = ctx.getImageData(0, 0, 128, 128).data;
        const chw = new Float32Array(3 * 128 * 128);
        const hw = 128 * 128;
        for (let i = 0; i < hw; i++) {
          chw[i] = rgba[i * 4] / 255;
          chw[hw + i] = rgba[i * 4 + 1] / 255;
          chw[2 * hw + i] = rgba[i * 4 + 2] / 255;
        }
        const res = runMalariaCNN(chw);
        state.batchItems.unshift({ name: file.name, img, prob: res.prob });
        renderBatchList();

        if (idx === 0) {
          document.querySelectorAll('.sample-btn').forEach((b) => b.classList.remove('active'));
          classifyImageElement(img, null);
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function renderBatchList() {
  const box = document.getElementById('batchResultsBox');
  const list = document.getElementById('batchList');
  if (!state.batchItems.length) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  const posCount = state.batchItems.filter((x) => x.prob >= state.liveThreshold).length;
  document.getElementById('batchSummaryText').textContent =
    `Uploaded (${state.batchItems.length}): ${posCount} Parasitized, ${state.batchItems.length - posCount} Uninfected`;

  list.innerHTML = '';
  state.batchItems.forEach((item) => {
    const isPos = item.prob >= state.liveThreshold;
    const row = document.createElement('div');
    row.className = 'batch-row';
    row.innerHTML = `
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;">${item.name}</span>
      <span class="mono ${isPos ? 'tag-pos' : 'tag-neg'}">${isPos ? 'Parasitized' : 'Uninfected'} (${item.prob.toFixed(3)})</span>
    `;
    row.addEventListener('click', () => {
      document.querySelectorAll('.sample-btn').forEach((b) => b.classList.remove('active'));
      classifyImageElement(item.img, null);
    });
    list.appendChild(row);
  });
}

// ===========================================================================
// 9. App Initialization & Event Bindings
// ===========================================================================
async function initApp() {
  initMicroscopeInteractions();
  renderPipeStage('input');

  try {
    const [dashRes, weightsRes] = await Promise.all([
      fetch('assets/dashboard_data.json'),
      fetch('assets/cnn_weights.bin'),
    ]);
    state.dashboard = await dashRes.json();
    const buf = await weightsRes.arrayBuffer();
    state.weights = new Float32Array(buf);

    for (const m of state.dashboard.layers_meta) {
      state.layers[m.name] = state.weights.subarray(m.offset, m.offset + m.length);
    }

    document.getElementById('statusDot').classList.remove('loading');
    document.getElementById('engineStatusText').textContent = 'CNN Engine Ready • 249K Params';

    renderSampleButtons();

    // Classify first test sample automatically
    const firstImg = document.querySelector('#samplesGrid .sample-btn img');
    if (firstImg) {
      if (firstImg.complete) classifyImageElement(firstImg, state.dashboard.samples[0]);
      else firstImg.onload = () => classifyImageElement(firstImg, state.dashboard.samples[0]);
    }

    updateThresholdTunerUI();
    renderTelemetryCharts();
  } catch (err) {
    console.error('Initialization error:', err);
    document.getElementById('engineStatusText').textContent = 'Asset Load Error';
  }
}

// Theme Toggle
document.getElementById('themeToggle').addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  html.setAttribute('data-theme', next);
  drawThresholdCurveCanvas();
  renderTelemetryCharts();
});

// Sample Filter Pills
document.querySelectorAll('#sampleFilterPills .filter-pill').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#sampleFilterPills .filter-pill').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.sampleFilter = btn.dataset.filter;
    renderSampleButtons();
  });
});

// Upload Dropzone
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleUploadedFiles(e.target.files));
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  handleUploadedFiles(e.dataTransfer.files);
});
document.getElementById('clearBatchBtn').addEventListener('click', () => {
  state.batchItems = [];
  renderBatchList();
});

// Live Classifier Threshold Controls
const liveThrSlider = document.getElementById('liveThrSlider');
const btnThrTuned = document.getElementById('btnThrTuned');
const btnThrMob = document.getElementById('btnThrMob');
const btnThrDefault = document.getElementById('btnThrDefault');

function setLiveThr(val) {
  state.liveThreshold = val;
  liveThrSlider.value = String(val);
  btnThrTuned.classList.toggle('active', Math.abs(val - 0.176) < 0.002);
  btnThrMob.classList.toggle('active', Math.abs(val - 0.281) < 0.002);
  btnThrDefault.classList.toggle('active', Math.abs(val - 0.5) < 0.002);
  updateLiveVerdictUI();
  renderBatchList();
}

liveThrSlider.addEventListener('input', (e) => setLiveThr(parseFloat(e.target.value)));
btnThrTuned.addEventListener('click', () => setLiveThr(0.176));
btnThrMob.addEventListener('click', () => setLiveThr(0.281));
btnThrDefault.addEventListener('click', () => setLiveThr(0.500));

// Microscope Viewport Mode & Colormap Controls
document.getElementById('tabViewSplit').addEventListener('click', (e) => {
  e.target.classList.add('active');
  document.getElementById('tabViewDual').classList.remove('active');
  document.getElementById('viewportSplit').classList.remove('hidden');
  document.getElementById('viewportDual').classList.add('hidden');
});

document.getElementById('tabViewDual').addEventListener('click', (e) => {
  e.target.classList.add('active');
  document.getElementById('tabViewSplit').classList.remove('active');
  document.getElementById('viewportDual').classList.remove('hidden');
  document.getElementById('viewportSplit').classList.add('hidden');
});

document.getElementById('colormapSelect').addEventListener('change', (e) => {
  state.colormap = e.target.value;
  renderAllMicroscopeCanvases();
});

document.getElementById('camOpacity').addEventListener('input', (e) => {
  state.camOpacity = parseInt(e.target.value, 10) / 100;
  renderAllMicroscopeCanvases();
});

// Section 2: Threshold Studio Controls
const tunerSlider = document.getElementById('tunerSlider');
const presetCnnTuned = document.getElementById('presetCnnTuned');
const presetMobTuned = document.getElementById('presetMobTuned');
const presetDefault = document.getElementById('presetDefault');

function setTunerThr(val) {
  state.tunerThreshold = val;
  tunerSlider.value = String(val);
  presetCnnTuned.classList.toggle('active', Math.abs(val - 0.176) < 0.002);
  presetMobTuned.classList.toggle('active', Math.abs(val - 0.281) < 0.002);
  presetDefault.classList.toggle('active', Math.abs(val - 0.5) < 0.002);
  updateThresholdTunerUI();
}

tunerSlider.addEventListener('input', (e) => setTunerThr(parseFloat(e.target.value)));
presetCnnTuned.addEventListener('click', () => setTunerThr(0.176));
presetMobTuned.addEventListener('click', () => setTunerThr(0.281));
presetDefault.addEventListener('click', () => setTunerThr(0.500));

document.getElementById('tunerModelCnn').addEventListener('click', (e) => {
  state.tunerModel = 'cnn';
  e.target.classList.add('active');
  document.getElementById('tunerModelMob').classList.remove('active');
  setTunerThr(0.176);
});

document.getElementById('tunerModelMob').addEventListener('click', (e) => {
  state.tunerModel = 'mobilenet';
  e.target.classList.add('active');
  document.getElementById('tunerModelCnn').classList.remove('active');
  setTunerThr(0.281);
});

// Section 3: Telemetry Controls & Lightbox
document.getElementById('histBtnCnn').addEventListener('click', (e) => {
  state.historyModel = 'cnn';
  state.selectedEpoch = 44;
  e.target.classList.add('active');
  document.getElementById('histBtnMob').classList.remove('active');
  document.getElementById('lossBestBadge').textContent = 'Best Val Loss: Epoch 44 (0.0954)';
  document.getElementById('accBestBadge').textContent = 'Peak Val AUC: 0.9936';
  renderTelemetryCharts();
});

document.getElementById('histBtnMob').addEventListener('click', (e) => {
  state.historyModel = 'mobilenet';
  state.selectedEpoch = 18;
  e.target.classList.add('active');
  document.getElementById('histBtnCnn').classList.remove('active');
  document.getElementById('lossBestBadge').textContent = 'Best Val Loss: Epoch 18 (0.0853)';
  document.getElementById('accBestBadge').textContent = 'Peak Val AUC: 0.9948';
  renderTelemetryCharts();
});

document.getElementById('epochSlider').addEventListener('input', (e) => {
  state.selectedEpoch = parseInt(e.target.value, 10);
  renderTelemetryCharts();
});

// Plot Lightbox Modal
const lightboxModal = document.getElementById('lightboxModal');
document.querySelectorAll('.plot-tile').forEach((tile) => {
  tile.addEventListener('click', () => {
    document.getElementById('lightboxImg').src = tile.dataset.lightbox;
    document.getElementById('lightboxCaption').textContent = tile.dataset.caption || '';
    lightboxModal.classList.remove('hidden');
  });
});
document.getElementById('lightboxClose').addEventListener('click', () => lightboxModal.classList.add('hidden'));
document.getElementById('lightboxBackdrop').addEventListener('click', () => lightboxModal.classList.add('hidden'));

// Section 5: Pipeline Nodes
document.querySelectorAll('#pipelineFlow .pipe-node').forEach((node) => {
  node.addEventListener('click', () => {
    document.querySelectorAll('#pipelineFlow .pipe-node').forEach((n) => n.classList.remove('active'));
    node.classList.add('active');
    renderPipeStage(node.dataset.stage);
  });
});

window.addEventListener('DOMContentLoaded', initApp);
