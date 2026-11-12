const $ = (selector) => document.querySelector(selector);
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const chemicalColor = (name) => token(`--chemical-${name}`) || token('--color-cool');
const SHORT_NAMES = {
  sensory_thalamus: 'Thalamus',
  sensory_cortex: 'Sensory ctx',
  insula: 'Insula',
  amygdala_bla: 'BLA threat',
  amygdala_bla_reward: 'BLA reward',
  amygdala_cea: 'CeA',
  bnst: 'BNST',
  hippocampus: 'Hippocampus',
  vmpfc: 'vmPFC',
  dacc: 'dACC',
  dlpfc: 'dlPFC',
  nucleus_accumbens: 'Accumbens',
  lateral_habenula: 'Habenula',
  hypothalamus: 'Hypothalamus',
  oxytocin_pvn: 'Oxytocin',
  pag: 'PAG',
  vta: 'VTA',
  locus_coeruleus: 'LC',
  raphe: 'Raphe',
  basal_forebrain: 'Basal forebr.',
  motor_brainstem: 'Brainstem',
};

function canvasContext(canvas, fallbackHeight) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(90, Math.round(canvas.clientHeight || fallbackHeight));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = `11px ${token('--font-body')}`;
  ctx.textBaseline = 'middle';
  return { ctx, width, height };
}

// Shade the stimulus window and label seconds along the bottom edge.
function drawTimeFrame(ctx, box, timeline) {
  const duration = timeline.duration_seconds;
  const [on, off] = timeline.stimulus_seconds;
  const x = (seconds) => box.left + (seconds / duration) * box.width;
  ctx.fillStyle = token('--color-cool-soft');
  ctx.fillRect(x(on), box.top, x(Math.min(off, duration)) - x(on), box.height);
  ctx.fillStyle = token('--color-muted');
  ctx.textAlign = 'center';
  for (let second = 0; second <= duration; second += 1) {
    ctx.fillText(
      `${second} s`,
      Math.min(box.left + box.width - 8, Math.max(box.left + 8, x(second))),
      box.top + box.height + 9,
    );
  }
  ctx.textAlign = 'left';
}

export function drawSpikeRaster(result) {
  const raster = result.spikes;
  const { ctx, width, height } = canvasContext($('#spike-raster'), 330);
  const box = {
    left: 78,
    top: 4,
    width: width - 78 - 8,
    height: height - 4 - 18,
  };
  const layout = raster.layout || [];
  const band = box.height / Math.max(1, layout.length);
  const offsets = [];
  let total = 0;
  layout.forEach((region) => {
    offsets.push(total);
    total += region.cells;
  });
  drawTimeFrame(ctx, box, result.timeline);
  layout.forEach((region, index) => {
    const y = box.top + index * band;
    ctx.fillStyle = token('--color-muted');
    ctx.fillText(SHORT_NAMES[region.id] || region.id, 0, y + band / 2);
    ctx.strokeStyle = token('--color-rule');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(box.left, y + band);
    ctx.lineTo(box.left + box.width, y + band);
    ctx.stroke();
  });
  const regionOf = (cell) => {
    let lo = 0,
      hi = offsets.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (offsets[mid] <= cell) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  };
  const durationMs = result.timeline.duration_seconds * 1000;
  for (const [timeMs, cell] of raster.events || []) {
    const index = regionOf(cell);
    const region = layout[index];
    const local = cell - offsets[index];
    const gaba = local >= region.projection || region.transmitter === 'gaba';
    ctx.fillStyle = gaba ? token('--color-inhibitory') : token('--color-cool');
    ctx.fillRect(
      box.left + (timeMs / durationMs) * box.width,
      box.top + index * band + ((local + 0.5) / region.cells) * band,
      1.2,
      1,
    );
  }
}

export function drawChemistryChart(result) {
  const { ctx, width, height } = canvasContext($('#chem-chart'), 170);
  const box = {
    left: 30,
    top: 6,
    width: width - 30 - 8,
    height: height - 6 - 18,
  };
  const course = result.chemistry.timecourse;
  const duration = result.timeline.duration_seconds;
  drawTimeFrame(ctx, box, result.timeline);
  const lo = 0,
    hi = 1;
  const y = (value) => box.top + (1 - (value - lo) / Math.max(0.01, hi - lo)) * box.height;
  ctx.fillStyle = token('--color-muted');
  [lo, (lo + hi) / 2, hi].forEach((value) => {
    ctx.fillText(value.toFixed(2), 0, y(value));
  });
  const legend = $('#chem-legend');
  legend.replaceChildren();
  for (const [name, series] of Object.entries(course.series)) {
    ctx.beginPath();
    ctx.strokeStyle = chemicalColor(name);
    ctx.lineWidth = 1.6;
    series.forEach((value, index) => {
      const x = box.left + (course.t[index] / duration) * box.width;
      if (index === 0) ctx.moveTo(x, y(value));
      else ctx.lineTo(x, y(value));
    });
    ctx.stroke();
    const key = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.style.background = chemicalColor(name);
    key.append(swatch, name);
    legend.append(key);
  }
}

export function drawEeg(result) {
  const eeg = result.eeg;
  const { ctx, width, height } = canvasContext($('#eeg-chart'), 205);
  const box = {
    left: 72,
    top: 4,
    width: width - 72 - 8,
    height: height - 4 - 18,
  };
  const channels = Object.entries(eeg.channels);
  const names = {
    frontal: 'Frontal-like',
    central: 'Central-like',
    posterior: 'Posterior-like',
  };
  const colors = [token('--color-cool'), token('--color-cool-2'), token('--color-inhibitory')];
  drawTimeFrame(ctx, box, result.timeline);
  const scale = sharedAmplitude(eeg.channels);
  $('#eeg-scale').textContent = `Shared scale: ±${scale.toPrecision(3)} a.u.`;
  channels.forEach(([name, values], channelIndex) => {
    if (!values.length) return;
    const bandHeight = box.height / channels.length;
    const baseline = box.top + channelIndex * bandHeight + bandHeight / 2;
    ctx.strokeStyle = token('--color-rule');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(box.left, baseline);
    ctx.lineTo(box.left + box.width, baseline);
    ctx.stroke();
    ctx.fillStyle = token('--color-muted');
    ctx.fillText(names[name] || name, 0, baseline);
    ctx.beginPath();
    ctx.strokeStyle = colors[channelIndex % colors.length];
    ctx.lineWidth = 1.1;
    values.forEach((value, index) => {
      const x = box.left + (index / Math.max(1, values.length - 1)) * box.width;
      const y = baseline - (value / scale) * bandHeight * 0.37;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

export function drawSpectrum(spectrum) {
  if (!spectrum) return;
  const { ctx, width, height } = canvasContext($('#eeg-spectrum'), 155);
  const frequencies = spectrum.frequencies_hz || [];
  const power = spectrum.power || [];
  const left = 30,
    right = 8,
    top = 8,
    bottom = 8;
  const plotWidth = width - left - right,
    plotHeight = height - top - bottom;
  [0, 0.5, 1].forEach((value) => {
    const y = top + plotHeight * (1 - value);
    ctx.strokeStyle = token('--color-rule');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
  });
  ctx.beginPath();
  ctx.strokeStyle = token('--color-cool-2');
  ctx.lineWidth = 1.6;
  frequencies.forEach((frequency, index) => {
    const x = left + ((frequency - 1) / 44) * plotWidth;
    const y = top + (1 - Math.max(0, Math.min(1, power[index] || 0))) * plotHeight;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  $('#peak-frequency').textContent = spectrum.peak_hz
    ? `Largest model peak: ${spectrum.peak_hz} Hz`
    : 'No clear peak';
  const bands = $('#eeg-bands');
  bands.replaceChildren();
  for (const [name, value] of Object.entries(spectrum.bands || {})) {
    const item = document.createElement('div');
    item.className = 'eeg-band';
    const label = document.createElement('span');
    label.textContent = name;
    const amount = document.createElement('b');
    amount.textContent = `${value}%`;
    item.append(label, amount);
    bands.append(item);
  }
}

// One symmetric amplitude scale preserves between-channel comparisons and every peak.
export function sharedAmplitude(channels) {
  let maximum = 0;
  for (const values of Object.values(channels))
    for (const value of values) maximum = Math.max(maximum, Math.abs(value));
  return maximum || 1;
}
