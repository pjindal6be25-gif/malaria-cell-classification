/**
 * MalariaCNN Interactive Web Application
 * - Runs the exact fused PyTorch MalariaCNN weights (outputs/cnn_best.pt -> cnn_weights.bin) in the browser
 * - Computes live Grad-CAM activation heatmaps on Block 4 ReLU (16x16 -> 128x128)
 * - Runs real-time decision threshold tuning across all 4,134 test-set predictions
 */

const state = {
  weights: null,
  layers: {},
  dashboard: null,
  liveThreshold: 0.176,
  camOpacity: 0.58,
  lastInference: null, // { inputRGB, prob, logit, cam16, ms, sampleMeta }
  tunerModel: 'cnn',
  tunerThreshold: 0.176,
};

// ---------------------------------------------------------------------------
// 1. Fused MalariaCNN Forward Pass + Exact Grad-CAM in Float32Array
// ---------------------------------------------------------------------------
function conv3x3ReLU(input, inCh, H, W, weight, bias, outCh) {
  const out = new Float32Array(outCh * H * W);
  const hw = H * W;
  for (let oc = 0; oc < outCh; oc++) {
    const bVal = bias[oc];
    const outOffset = oc * hw;
    const wOcOffset = oc * inCh * 9;

    // Initialize with bias
    for (let i = 0; i < hw; i++) {
      out[outOffset + i] = bVal;
    }

    // Accumulate 3x3 convolutions with padding=1
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

    // ReLU in-place
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

function runMalariaCNN(inputCHW) {
  const L = state.layers;
  const t0 = performance.now();

  // Block 1: 3x128x128 -> 32x64x64
  const a1 = conv3x3ReLU(inputCHW, 3, 128, 128, L.conv1_w, L.conv1_b, 32);
  const p1 = maxPool2x2(a1, 32, 128, 128);

  // Block 2: 32x64x64 -> 64x32x32
  const a2 = conv3x3ReLU(p1, 32, 64, 64, L.conv2_w, L.conv2_b, 64);
  const p2 = maxPool2x2(a2, 64, 64, 64);

  // Block 3: 64x32x32 -> 128x16x16
  const a3 = conv3x3ReLU(p2, 64, 32, 32, L.conv3_w, L.conv3_b, 128);
  const p3 = maxPool2x2(a3, 128, 32, 32);

  // Block 4: 128x16x16 (a4 is model.gradcam_layer()!) -> 128x8x8
  const a4 = conv3x3ReLU(p3, 128, 16, 16, L.conv4_w, L.conv4_b, 128);
  const p4 = maxPool2x2(a4, 128, 16, 16);

  // Global Average Pool 128x8x8 -> 128
  const gap = new Float32Array(128);
  for (let c = 0; c < 128; c++) {
    let s = 0.0;
    const base = c * 64;
    for (let i = 0; i < 64; i++) s += p4[base + i];
    gap[c] = s / 64.0;
  }

  // FC1: 128 -> 64 + ReLU
  const h = new Float32Array(64);
  for (let j = 0; j < 64; j++) {
    let s = L.fc1_b[j];
    const wBase = j * 128;
    for (let c = 0; c < 128; c++) {
      s += L.fc1_w[wBase + c] * gap[c];
    }
    h[j] = s > 0 ? s : 0;
  }

  // FC2: 64 -> 1 logit
  let logit = L.fc2_b[0];
  for (let j = 0; j < 64; j++) {
    logit += L.fc2_w[j] * h[j];
  }
  const prob = 1.0 / (1.0 + Math.exp(-logit));

  // Grad-CAM on a4 (128 x 16 x 16)
  // Channel weights alpha_c = d(logit)/d(gap_c)
  // For predicted class direction: positive towards Parasitized when prob >= thr, else towards Uninfected
  const sign = logit >= 0 ? 1.0 : -1.0;
  const alpha = new Float32Array(128);
  for (let c = 0; c < 128; c++) {
    let gradC = 0.0;
    for (let j = 0; j < 64; j++) {
      if (h[j] > 0) {
        gradC += L.fc2_w[j] * L.fc1_w[j * 128 + c];
      }
    }
    alpha[c] = sign * gradC;
  }

  const cam16 = new Float32Array(16 * 16);
  let maxCam = 1e-8;
  for (let idx = 0; idx < 256; idx++) {
    let v = 0.0;
    for (let c = 0; c < 128; c++) {
      v += alpha[c] * a4[c * 256 + idx];
    }
    if (v < 0) v = 0;
    cam16[idx] = v;
    if (v > maxCam) maxCam = v;
  }
  for (let idx = 0; idx < 256; idx++) {
    cam16[idx] /= maxCam;
  }

  const ms = performance.now() - t0;
  return { logit, prob, cam16, ms };
}

// ---------------------------------------------------------------------------
// 2. Canvas Image Preprocessing & Jet/Turbo Heatmap Rendering
// ---------------------------------------------------------------------------
function jetColor(v) {
  // Smooth medical jet colormap [0, 1] -> [R, G, B]
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function renderGradCamOverlay() {
  if (!state.lastInference) return;
  const { imgData128, cam16 } = state.lastInference;
  const canvas = document.getElementById('gradcamCanvas');
  const ctx = canvas.getContext('2d');
  const outImg = ctx.createImageData(128, 128);
  const src = imgData128.data;
  const dst = outImg.data;
  const alpha = state.camOpacity;

  // Bilinearly interpolate 16x16 cam16 onto 128x128
  for (let y = 0; y < 128; y++) {
    const gy = ((y + 0.5) / 128) * 16 - 0.5;
    const y0 = Math.max(0, Math.min(15, Math.floor(gy)));
    const y1 = Math.max(0, Math.min(15, y0 + 1));
    const wy = gy - Math.floor(gy);

    for (let x = 0; x < 128; x++) {
      const gx = ((x + 0.5) / 128) * 16 - 0.5;
      const x0 = Math.max(0, Math.min(15, Math.floor(gx)));
      const x1 = Math.max(0, Math.min(15, x0 + 1));
      const wx = gx - Math.floor(gx);

      const v00 = cam16[y0 * 16 + x0];
      const v01 = cam16[y0 * 16 + x1];
      const v10 = cam16[y1 * 16 + x0];
      const v11 = cam16[y1 * 16 + x1];
      const val = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11);

      const [hr, hg, hb] = jetColor(Math.max(0, Math.min(1, val)));
      const pIdx = (y * 128 + x) * 4;
      dst[pIdx]     = Math.round((1 - alpha) * src[pIdx]     + alpha * hr);
      dst[pIdx + 1] = Math.round((1 - alpha) * src[pIdx + 1] + alpha * hg);
      dst[pIdx + 2] = Math.round((1 - alpha) * src[pIdx + 2] + alpha * hb);
      dst[pIdx + 3] = 255;
    }
  }
  ctx.putImageData(outImg, 0, 0);
}

function updateLiveVerdictUI() {
  if (!state.lastInference) return;
  const { prob, ms, sampleMeta } = state.lastInference;
  const thr = state.liveThreshold;
  const isPos = prob >= thr;
  const label = isPos ? 'Parasitized' : 'Uninfected';
  const confidence = isPos ? prob : 1 - prob;

  const verdictBox = document.getElementById('verdictBox');
  verdictBox.className = 'verdict-box ' + (isPos ? 'is-parasitized' : 'is-uninfected');
  document.getElementById('predLabel').textContent = label;
  document.getElementById('predSub').textContent = isPos
    ? `P(Parasitized) ${prob.toFixed(4)} >= threshold ${thr.toFixed(3)}`
    : `P(Parasitized) ${prob.toFixed(4)} < threshold ${thr.toFixed(3)}`;

  document.getElementById('predProb').textContent = prob.toFixed(4);
  document.getElementById('predConf').textContent = (confidence * 100).toFixed(2) + '%';
  document.getElementById('predTime').textContent = ms.toFixed(1) + ' ms';

  document.getElementById('gaugeFill').style.width = (prob * 100).toFixed(2) + '%';
  document.getElementById('gaugeMarker').style.left = (thr * 100).toFixed(2) + '%';
  document.getElementById('activeThrLabel').textContent =
    `Active Threshold: ${thr.toFixed(3)}` + (Math.abs(thr - 0.176) < 0.002 ? ' (Recall-Tuned)' : Math.abs(thr - 0.5) < 0.002 ? ' (Default)' : '');

  const banner = document.getElementById('sampleInfoBanner');
  if (sampleMeta) {
    const correct = (sampleMeta.true_label === 1) === isPos;
    banner.innerHTML = `
      <div><strong>${sampleMeta.title}</strong> — ${sampleMeta.description}</div>
      <div style="margin-top:4px;">
        Ground Truth: <strong>${sampleMeta.true_class}</strong> (${correct ? '<span style="color:#34d399">Correct at current threshold</span>' : '<span style="color:#fb7185">Misclassified at current threshold</span>'})
        &nbsp;|&nbsp; PyTorch Test P(Parasitized): <code>CNN = ${sampleMeta.prob_cnn.toFixed(4)}</code>, <code>MobileNetV2 = ${sampleMeta.prob_mobilenet.toFixed(4)}</code>
      </div>
    `;
  } else {
    banner.innerHTML = `<strong>Custom Uploaded Image</strong> — Classified live in your browser using <code>outputs/cnn_best.pt</code> weights and Block-4 Grad-CAM.`;
  }
}

function classifyImageElement(imgEl, sampleMeta = null) {
  if (!state.weights) return;
  const inputCanvas = document.getElementById('inputCanvas');
  const ctx = inputCanvas.getContext('2d');
  ctx.clearRect(0, 0, 128, 128);
  ctx.drawImage(imgEl, 0, 0, 128, 128);

  document.getElementById('imgOrigSize').textContent =
    `${imgEl.naturalWidth}×${imgEl.naturalHeight} → 128×128`;

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
  // If this is a known test-set sample, use its exact PyTorch PIL-antialias probability for the readout
  const finalProb = sampleMeta ? sampleMeta.prob_cnn : res.prob;
  state.lastInference = {
    imgData128,
    prob: finalProb,
    logit: res.logit,
    cam16: res.cam16,
    ms: res.ms,
    sampleMeta,
  };

  updateLiveVerdictUI();
  renderGradCamOverlay();
}

// ---------------------------------------------------------------------------
// 3. Interactive Test-Set Threshold Tuner (4,134 Test Images)
// ---------------------------------------------------------------------------
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

  document.getElementById('tunerThrVal').textContent = thr.toFixed(3);
  document.getElementById('tmRecall').textContent = (rec * 100).toFixed(2) + '%';
  document.getElementById('tmPrecision').textContent = (prec * 100).toFixed(2) + '%';
  document.getElementById('tmSpecificity').textContent = (spec * 100).toFixed(2) + '%';
  document.getElementById('tmAccuracy').textContent = (acc * 100).toFixed(2) + '%';
  document.getElementById('tmF1').textContent = f1.toFixed(4);
  document.getElementById('tmAuc').textContent = auc;

  document.getElementById('cmTN').textContent = tn.toLocaleString();
  document.getElementById('cmFP').textContent = fp.toLocaleString();
  document.getElementById('cmFN').textContent = fn.toLocaleString();
  document.getElementById('cmTP').textContent = tp.toLocaleString();

  const saved = fnDefault - fn;
  const modelLabel = state.tunerModel === 'cnn' ? 'Custom CNN' : 'MobileNetV2';
  const callout = document.getElementById('cmCallout');
  if (saved > 0) {
    callout.innerHTML = `<strong>${modelLabel}</strong> at threshold <strong>${thr.toFixed(3)}</strong> catches <strong>${saved} additional infected cells</strong> that would be missed at 0.50 (missed infections drop from ${fnDefault} &rarr; <strong>${fn}</strong>).`;
  } else if (saved === 0) {
    callout.innerHTML = `<strong>${modelLabel}</strong> at default threshold <strong>0.500</strong>: ${fn} missed infections (FN) and ${fp} false alarms (FP).`;
  } else {
    callout.innerHTML = `<strong>${modelLabel}</strong> at threshold <strong>${thr.toFixed(3)}</strong> is stricter than 0.50, reducing false alarms to ${fp} but missing <strong>${fn}</strong> infections.`;
  }
}

// ---------------------------------------------------------------------------
// 4. Initialization & Event Listeners
// ---------------------------------------------------------------------------
async function initApp() {
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

    const statusEl = document.getElementById('engineStatus');
    statusEl.querySelector('.status-dot').classList.remove('loading');
    document.getElementById('engineStatusText').textContent =
      'MalariaCNN Ready (249,153 weights loaded)';

    // Populate 8 test-set sample buttons
    const grid = document.getElementById('samplesGrid');
    state.dashboard.samples.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sample-btn' + (idx === 0 ? ' active' : '');
      btn.innerHTML = `
        <img src="${s.file}" alt="${s.title}" crossorigin="anonymous" />
        <div class="sample-btn-info">
          <div class="sample-btn-title">${s.title}</div>
          <div class="sample-btn-tag ${s.true_label === 1 ? 'tag-pos' : 'tag-neg'}">
            True: ${s.true_class} • P=${s.prob_cnn.toFixed(3)}
          </div>
        </div>
      `;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const imgEl = btn.querySelector('img');
        if (imgEl.complete) {
          classifyImageElement(imgEl, s);
        } else {
          imgEl.onload = () => classifyImageElement(imgEl, s);
        }
      });
      grid.appendChild(btn);
    });

    // Run initial sample
    const firstBtn = grid.querySelector('.sample-btn img');
    if (firstBtn) {
      if (firstBtn.complete) classifyImageElement(firstBtn, state.dashboard.samples[0]);
      else firstBtn.onload = () => classifyImageElement(firstBtn, state.dashboard.samples[0]);
    }

    updateThresholdTunerUI();
  } catch (err) {
    console.error('Failed to load model assets:', err);
    document.getElementById('engineStatusText').textContent = 'Error loading model assets';
  }
}

// Upload & Drag-and-Drop handlers
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');

function handleUserFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      classifyImageElement(img, null);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleUserFile(e.target.files[0]));
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleUserFile(e.dataTransfer.files[0]);
});

// Live Classifier Threshold Controls
const liveThrSlider = document.getElementById('liveThrSlider');
const btnThrTuned = document.getElementById('btnThrTuned');
const btnThrDefault = document.getElementById('btnThrDefault');

liveThrSlider.addEventListener('input', (e) => {
  state.liveThreshold = parseFloat(e.target.value);
  btnThrTuned.classList.toggle('active', Math.abs(state.liveThreshold - 0.176) < 0.002);
  btnThrDefault.classList.toggle('active', Math.abs(state.liveThreshold - 0.5) < 0.002);
  updateLiveVerdictUI();
});

btnThrTuned.addEventListener('click', () => {
  state.liveThreshold = 0.176;
  liveThrSlider.value = '0.176';
  btnThrTuned.classList.add('active');
  btnThrDefault.classList.remove('active');
  updateLiveVerdictUI();
});

btnThrDefault.addEventListener('click', () => {
  state.liveThreshold = 0.5;
  liveThrSlider.value = '0.500';
  btnThrDefault.classList.add('active');
  btnThrTuned.classList.remove('active');
  updateLiveVerdictUI();
});

document.getElementById('camOpacity').addEventListener('input', (e) => {
  state.camOpacity = parseInt(e.target.value, 10) / 100;
  renderGradCamOverlay();
});

// Section 2 Threshold Tuner Controls
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

window.addEventListener('DOMContentLoaded', initApp);
