import {
  chemicalColor,
  drawChemistryChart,
  drawEeg,
  drawSpectrum,
  drawSpikeRaster,
} from './charts.mjs';
import { createMemoryStore, validateMemory } from './memory.mjs';
import { mountWorkbench } from './workbench.mjs';

const memoryStore = createMemoryStore(localStorage);
/** @param {string} selector @param {ParentNode} root @returns {any} */
const $ = (selector, root = document) => root.querySelector(selector);
/** @param {string} selector @param {ParentNode} root @returns {any[]} */
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
let memory = memoryStore.value;
let latest = null;
let pendingPhoto = null;
let modelInfo = null;
const photoPreview = document.createElement('img');
photoPreview.id = 'photo-preview';
photoPreview.alt = 'Local photo preview';
$('#photo-preview').replaceWith(photoPreview);
const workbench = mountWorkbench(() => {
  if (!latest) return;
  drawSpikeRaster(latest.result);
  drawChemistryChart(latest.result);
  drawEeg(latest.result);
  drawSpectrum(latest.result.eeg?.spectrum);
});

const PRESETS = {
  threat: { threat_cue: 0.9, intensity: 0.4 },
  reward: { reward_cue: 0.9 },
  loss: { loss_cue: 0.9 },
  pain: { pain: 0.8 },
  warm: { social_touch: 0.9 },
  novel: { novelty: 0.9 },
  clear: {},
};
function persistMemory(candidate) {
  try {
    memory = memoryStore.commit(candidate);
  } catch (error) {
    $('#memory-status').textContent = `Memory was not changed: ${error.message}`;
    return false;
  }
  $('#memory-status').textContent = memoryStore.warning;
  renderMemory();
  return true;
}
function normalizedRange(input) {
  return Number(input.value) / 100;
}
function personById(id) {
  return memory.people.find((person) => person.id === id);
}
function signed(value, digits = 2) {
  const number = Number(value) || 0;
  return `${number >= 0 ? '+' : '−'}${Math.abs(number).toFixed(digits)}`;
}

function bindRange(input) {
  const output = input.parentElement.querySelector('output');
  const update = () => {
    if (output) output.value = input.value;
  };
  input.addEventListener('input', update);
  update();
}
$$('input[type="range"]').forEach(bindRange);

$('#event-text').addEventListener('input', (event) => {
  $('#char-count').textContent = `${event.target.value.length} / 500`;
});

function refreshPersonSelect(selected = $('#person-select').value) {
  const select = $('#person-select');
  select.replaceChildren(new Option('No person selected', ''));
  for (const person of memory.people) {
    const option = new Option(
      `${person.name}${person.relationship ? ` / ${person.relationship}` : ''}`,
      person.id,
    );
    select.add(option);
  }
  if (personById(selected)) select.value = selected;
  updatePersonContext();
}
function updatePersonContext() {
  const person = personById($('#person-select').value);
  $('#person-context').textContent = person
    ? 'Trust, closeness and tone become numeric social cues. Name and notes stay here.'
    : '';
}
$('#person-select').addEventListener('change', updatePersonContext);

// ---------------------------------------------------------------- controls built from the network definition

function sliderRow(channel) {
  const row = document.createElement('label');
  row.className = 'slider-row';
  const name = document.createElement('span');
  const strong = document.createElement('strong');
  strong.textContent = channel.label;
  name.append(strong);
  const input = document.createElement('input');
  Object.assign(input, {
    type: 'range',
    min: 0,
    max: 100,
    value: Math.round(channel.default * 100),
  });
  input.dataset.key = channel.id;
  input.dataset.default = String(Math.round(channel.default * 100));
  const output = document.createElement('output');
  row.append(name, input, output);
  bindRange(input);
  return row;
}

async function loadModelInfo() {
  try {
    const response = await fetch('/api/info', { cache: 'no-store' });
    if (!response.ok) throw new Error('Model information unavailable.');
    modelInfo = await response.json();
  } catch {
    $('#hardware-status').textContent = 'The local server is not responding. Restart the app.';
    $('#run-button').disabled = true;
    return;
  }
  const groups = {
    senses: $('#sense-sliders'),
    body: $('#body-sliders'),
    mind: $('#mind-sliders'),
  };
  Object.values(groups).forEach((group) => {
    group.replaceChildren();
  });
  for (const channel of modelInfo.inputs) groups[channel.group]?.append(sliderRow(channel));
  const [on, off] = modelInfo.timeline.stimulus_seconds;
  $$('.stimulus-on').forEach((node) => {
    node.textContent = on;
  });
  $$('.stimulus-off').forEach((node) => {
    node.textContent = off;
  });

  const select = $('#model-size');
  select.replaceChildren(
    ...(modelInfo.network_sizes || []).map(
      (size) => new Option(`${size.toLocaleString()} cells`, size),
    ),
  );
  if (modelInfo.default_neurons) select.value = String(modelInfo.default_neurons);
  const device = modelInfo.hardware || {};
  $('#run-button').disabled = !modelInfo.available;
  $('#create-session').disabled = !modelInfo.available;
  if (!modelInfo.available) {
    $('#hardware-status').textContent = `Cannot run: ${device.name}`;
    $('#run-button').disabled = true;
  } else if (device.available) {
    $('#hardware-status').textContent =
      `${device.name} / ${device.vram_gib} GiB VRAM / CUDA active. Larger networks fall back to the largest size that fits in free memory.`;
  } else {
    $('#hardware-status').textContent =
      `No CUDA GPU found / running on the ${device.name}. The same network runs, limited to smaller sizes.`;
  }
}
loadModelInfo();

$$('.preset-button').forEach((button) => {
  button.addEventListener('click', () => {
    const values = PRESETS[button.dataset.preset] || {};
    $$('#sense-sliders input[data-key]').forEach((input) => {
      input.value = Math.round((values[input.dataset.key] || 0) * 100);
      input.dispatchEvent(new Event('input'));
    });
  });
});

function getRequest() {
  const inputs = {};
  $$('input[type="range"][data-key]').forEach((input) => {
    inputs[input.dataset.key] = normalizedRange(input);
  });
  const person = personById($('#person-select').value);
  const request = {
    event: $('#event-text').value.trim(),
    inputs,
    neurons: Number($('#model-size').value),
  };
  if (person) {
    request.person = {
      trust: Number(person.trust) || 0,
      closeness: Number(person.closeness) || 0,
      tone: Number(person.valence ?? 0.5),
    };
  }
  const seed = $('#seed-input').value.trim();
  if (seed !== '') request.seed = Number(seed);
  return { request, person };
}

$('#run-button').addEventListener('click', async () => {
  const button = $('#run-button');
  const { request: localRequest, person } = getRequest();
  const eventNote = localRequest.event;
  delete localRequest.event;
  const request = workbench.request(localRequest);
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  $('#results').setAttribute('aria-busy', 'true');
  button.querySelector('span').textContent = `Running ${request.neurons.toLocaleString()} cells…`;
  $('#run-status').textContent =
    'The first run at a new size also builds the synapses and a resting reference, so it takes longer.';
  try {
    const response = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Simulation failed.');
    latest = {
      result,
      event: eventNote,
      inputs: request.inputs,
      personId: person?.id || '',
      personName: person?.name || '',
    };
    renderResult(result);
    await workbench.completed(result);
    $('#results').scrollIntoView({
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      block: 'start',
    });
  } catch (error) {
    $('#run-status').textContent = `Could not run: ${error.message}`;
  } finally {
    button.disabled = false;
    button.removeAttribute('aria-busy');
    $('#results').removeAttribute('aria-busy');
    button.querySelector('span').textContent = 'Run simulation';
  }
});

// ---------------------------------------------------------------- results

function setSignedMeter(meter, value) {
  const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
  meter.style.left = `${50 + Math.min(0, clamped) * 50}%`;
  meter.style.width = `${Math.abs(clamped) * 50}%`;
}

function renderResult(result) {
  $('#empty-result').hidden = true;
  $('#result-content').hidden = false;
  const sim = result.simulation;
  $('#signal-summary').textContent =
    `${Number(result.spikes.total_count).toLocaleString()} spikes / ${sim.neuron_count.toLocaleString()} cells / ` +
    `${sim.synapse_count.toLocaleString()} synapses / ${sim.duration_seconds} s / ${sim.device} / seed ${sim.random_seed}`;

  const emotions = result.emotions || [];
  $('#top-emotion').textContent = emotions[0]?.label || 'Mixed pattern';
  $('#emotion-chips').replaceChildren(
    ...emotions.map((item, index) => {
      const chip = document.createElement('span');
      chip.className = 'emotion-chip';
      const label = document.createElement('span');
      label.textContent = `${index === 0 ? item.label : `Also ${item.label}`} `;
      const score = document.createElement('b');
      score.textContent = `${Math.round(item.score)}%`;
      chip.append(label, score);
      return chip;
    }),
  );
  const dims = result.dimensions;
  for (const [key, id] of [
    ['valence', 'valence'],
    ['arousal', 'arousal'],
    ['agency', 'agency'],
    ['social_safety', 'safety'],
  ]) {
    $(`#${id}-value`).textContent = signed(dims[key]);
    setSignedMeter($(`#${id}-meter`), dims[key]);
  }

  drawSpikeRaster(result);
  drawChemistryChart(result);
  renderChemistry(result.chemistry);
  drawEeg(result);
  drawSpectrum(result.eeg?.spectrum);
  $('#interpretation-text').textContent = result.interpretation;
  $('#result-caveat').textContent = result.caveat;
  renderRegions(result.regions);

  const requested = sim.requested_neuron_count,
    actual = sim.neuron_count;
  $('#run-status').textContent =
    (actual < requested
      ? `Used ${actual.toLocaleString()} cells because less memory was free. `
      : '') +
    `Finished in ${sim.elapsed_seconds}s on ${sim.device}. Enter seed ${sim.random_seed} to repeat this run.`;
}

function renderChemistry(chemistry) {
  const container = $('#chemistry-readout');
  container.replaceChildren();
  const row = (name, fraction, tick, text) => {
    const item = document.createElement('div');
    item.className = 'chemistry-row';
    const label = document.createElement('span');
    label.textContent = name;
    const track = document.createElement('div');
    track.className = 'chemistry-track';
    const fill = document.createElement('span');
    fill.style.width = `${Math.max(0, Math.min(100, fraction * 100))}%`;
    fill.style.background = chemicalColor(name.toLowerCase());
    const marker = document.createElement('i');
    marker.style.left = `${Math.max(0, Math.min(100, tick * 100))}%`;
    track.append(fill, marker);
    const value = document.createElement('b');
    value.textContent = text;
    item.append(label, track, value);
    container.append(item);
  };
  for (const chem of chemistry.neuromodulators) {
    const name = chem.name[0].toUpperCase() + chem.name.slice(1);
    row(
      name,
      chem.response,
      chem.baseline,
      `${chem.baseline.toFixed(2)} → ${chem.response.toFixed(2)}`,
    );
  }
  const release = chemistry.synaptic_release_per_cell_hz;
  const peak =
    Math.max(release.glutamate[1], release.gaba[1], release.glutamate[0], release.gaba[0], 0.1) *
    1.2;
  row(
    'Glutamate',
    release.glutamate[1] / peak,
    release.glutamate[0] / peak,
    `${release.glutamate[0].toFixed(1)} → ${release.glutamate[1].toFixed(1)}/s`,
  );
  row(
    'GABA',
    release.gaba[1] / peak,
    release.gaba[0] / peak,
    `${release.gaba[0].toFixed(1)} → ${release.gaba[1].toFixed(1)}/s`,
  );
}

function renderRegions(regions) {
  const container = $('#region-list');
  container.replaceChildren();
  const sorted = [...regions].sort((a, b) => Math.abs(b.change_log2) - Math.abs(a.change_log2));
  const maxRate = Math.max(
    1,
    ...regions.map((region) => Math.max(region.response_hz, region.baseline_hz)),
  );
  sorted.forEach((region, index) => {
    const row = document.createElement('div');
    row.className = 'region-row';
    row.hidden = index > 5;
    row.innerHTML =
      '<div><div class="region-name"></div><div class="region-role"></div></div><div class="region-track"><div class="region-fill"></div><i></i></div><div class="region-value"></div>';
    $('.region-name', row).textContent = region.label;
    $('.region-role', row).textContent =
      `${region.role} ${region.cells.toLocaleString()} cells / ${region.transmitter === 'gaba' ? 'GABA' : 'glutamate'} output.`;
    $('.region-fill', row).style.width = `${Math.round((100 * region.response_hz) / maxRate)}%`;
    $('.region-fill', row).classList.toggle('gaba', region.transmitter === 'gaba');
    $('.region-track i', row).style.left = `${(100 * region.baseline_hz) / maxRate}%`;
    $('.region-value', row).textContent =
      `${region.baseline_hz.toFixed(1)} → ${region.response_hz.toFixed(1)} Hz`;
    container.append(row);
  });
  $('#region-toggle').hidden = regions.length <= 6;
  $('#region-toggle').textContent = 'Show all';
  $('#region-toggle').setAttribute('aria-expanded', 'false');
}
$('#region-toggle').addEventListener('click', () => {
  const expanded = $('#region-toggle').getAttribute('aria-expanded') === 'true';
  $$('.region-row', $('#region-list')).forEach((row, index) => {
    row.hidden = expanded && index > 5;
  });
  $('#region-toggle').setAttribute('aria-expanded', String(!expanded));
  $('#region-toggle').textContent = expanded ? 'Show all' : 'Show less';
});

// ---------------------------------------------------------------- charts

window.addEventListener('resize', () => {
  if (!latest) return;
  drawSpikeRaster(latest.result);
  drawChemistryChart(latest.result);
  drawEeg(latest.result);
  drawSpectrum(latest.result.eeg?.spectrum);
});

$('#save-episode').addEventListener('click', () => {
  if (!latest) return;
  const sim = latest.result.simulation;
  const entry = {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    savedAt: new Date().toISOString(),
    event: latest.event,
    personId: latest.personId,
    personName: latest.personName,
    topLabel: latest.result.emotions?.[0]?.label || 'Mixed pattern',
    dimensions: latest.result.dimensions,
    inputs: latest.inputs,
    seed: sim.random_seed,
    neurons: sim.neuron_count,
    modelVersion: 4,
  };
  if (
    persistMemory({
      ...memory,
      episodes: [entry, ...memory.episodes].slice(0, 250),
    })
  )
    $('#run-status').textContent = `Episode saved (seed ${sim.random_seed}).`;
});

function resetPersonForm() {
  $('#person-form').reset();
  $('#person-edit-id').value = '';
  pendingPhoto = null;
  $('#person-trust').value = 50;
  $('#person-closeness').value = 50;
  $('#person-valence').value = 50;
  $$('.person-sliders output').forEach((output) => {
    output.value = output.previousElementSibling?.value || '50';
  });
  $('#save-person').textContent = 'Save person';
  $('#cancel-edit').hidden = true;
  $('#photo-preview-wrap').hidden = true;
  $('#photo-preview').removeAttribute('src');
  $('#person-photo').value = '';
}
$('#cancel-edit').addEventListener('click', resetPersonForm);

$('#person-photo').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    pendingPhoto = null;
    return;
  }
  if (!file.type.startsWith('image/')) {
    alert('Choose an image file.');
    event.target.value = '';
    return;
  }
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 420 / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    pendingPhoto = canvas.toDataURL('image/jpeg', 0.68);
    $('#photo-preview').src = pendingPhoto;
    $('#photo-preview-wrap').hidden = false;
  } catch {
    alert('This picture could not be opened.');
    event.target.value = '';
    pendingPhoto = null;
  }
});
$('#remove-photo').addEventListener('click', () => {
  pendingPhoto = '';
  $('#photo-preview-wrap').hidden = true;
  $('#person-photo').value = '';
});

$('#person-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const id = $('#person-edit-id').value;
  const existing = id ? personById(id) : null;
  const profile = {
    id: id || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`),
    name: $('#person-name').value.trim(),
    relationship: $('#person-relationship').value.trim(),
    meaning: $('#person-meaning').value.trim(),
    trust: normalizedRange($('#person-trust')),
    closeness: normalizedRange($('#person-closeness')),
    valence: normalizedRange($('#person-valence')),
    photoDataUrl: pendingPhoto === '' ? null : pendingPhoto || existing?.photoDataUrl || null,
    notes: existing?.notes || '',
    createdAt: existing?.createdAt || new Date().toISOString(),
  };
  if (!profile.name || !profile.meaning) return;
  const people = existing
    ? memory.people.map((person) => (person.id === id ? profile : person))
    : [profile, ...memory.people];
  if (persistMemory({ ...memory, people })) {
    refreshPersonSelect(profile.id);
    resetPersonForm();
    $('#person-context').textContent =
      `${profile.name} was saved. Select them above to use them as a social cue; the meaning note stays as your own memory.`;
  }
});

function editPerson(person) {
  $('#person-edit-id').value = person.id;
  $('#person-name').value = person.name || '';
  $('#person-relationship').value = person.relationship || '';
  $('#person-meaning').value = person.meaning || '';
  $('#person-trust').value = Math.round(Number(person.trust || 0) * 100);
  $('#person-closeness').value = Math.round(Number(person.closeness || 0) * 100);
  $('#person-valence').value = Math.round(Number(person.valence ?? 0.5) * 100);
  $$('.person-sliders input').forEach((input) => {
    input.dispatchEvent(new Event('input'));
  });
  pendingPhoto = person.photoDataUrl || null;
  $('#person-photo').value = '';
  if (pendingPhoto) {
    $('#photo-preview').src = pendingPhoto;
    $('#photo-preview-wrap').hidden = false;
  } else $('#photo-preview-wrap').hidden = true;
  $('#save-person').textContent = 'Update person';
  $('#cancel-edit').hidden = false;
  $('#person-name').focus();
}

function renderMemory() {
  $('#memory-count').textContent = `${memory.people.length + memory.episodes.length} saved`;
  $('#episode-count').textContent = `${memory.episodes.length} saved`;
  const peopleList = $('#people-list');
  peopleList.replaceChildren();
  for (const person of memory.people) {
    const card = $('#person-card-template').content.firstElementChild.cloneNode(true);
    const avatar = $('.person-avatar', card);
    if (person.photoDataUrl) {
      const image = document.createElement('img');
      image.src = person.photoDataUrl;
      image.alt = `Reference photo for ${person.name}`;
      avatar.append(image);
    } else avatar.textContent = (person.name || '?').trim().slice(0, 1).toUpperCase();
    $('h4', card).textContent = person.name || 'Unnamed person';
    $('.relationship-pill', card).textContent = person.relationship || 'person';
    $('.person-meaning', card).textContent = person.meaning || '';
    $('.person-stats', card).textContent =
      `trust ${Math.round((person.trust || 0) * 100)} / closeness ${Math.round((person.closeness || 0) * 100)}`;
    $('.edit-person', card).addEventListener('click', () => editPerson(person));
    $('.delete-person', card).addEventListener('click', () => {
      if (!confirm(`Delete ${person.name} and keep episodes that mention them?`)) return;
      persistMemory({
        ...memory,
        people: memory.people.filter((row) => row.id !== person.id),
      });
    });
    peopleList.append(card);
  }
  const episodes = $('#episodes-list');
  episodes.replaceChildren();
  memory.episodes.slice(0, 30).forEach((episode) => {
    const card = $('#episode-template').content.firstElementChild.cloneNode(true);
    const date = new Date(episode.savedAt);
    $('.episode-date', card).textContent = Number.isNaN(date.getTime())
      ? 'Saved'
      : date.toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
        });
    $('h4', card).textContent = episode.event || 'Saved episode';
    $('.episode-person', card).textContent = episode.personName
      ? `With ${episode.personName}`
      : 'No person selected';
    const dims = episode.dimensions || {};
    $('.episode-pattern', card).textContent =
      `${episode.topLabel || 'Pattern'} / valence ${signed(dims.valence)} / arousal ${signed(dims.arousal)}${episode.seed != null ? ` / seed ${episode.seed}` : ''}`;
    $('.delete-episode', card).addEventListener('click', () => {
      persistMemory({
        ...memory,
        episodes: memory.episodes.filter((row) => row.id !== episode.id),
      });
    });
    episodes.append(card);
  });
  refreshPersonSelect($('#person-select').value);
}

$('#export-memory').addEventListener('click', () => {
  const blob = new Blob(
    [
      JSON.stringify(
        {
          format: 'human-emotion-memory',
          version: 1,
          exportedAt: new Date().toISOString(),
          ...memory,
        },
        null,
        2,
      ),
    ],
    { type: 'application/json' },
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `human-emotion-memory-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('#import-memory').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (file.size > 15_000_000) throw new Error('File is larger than 15 MB.');
    const value = validateMemory(JSON.parse(await file.text()));
    if (!Array.isArray(value.people) || !Array.isArray(value.episodes))
      throw new Error('This file does not look like a Human Emotion memory backup.');
    if (
      !confirm(
        `Import ${value.people.length} people and ${value.episodes.length} episodes? This replaces the memory in this browser.`,
      )
    )
      return;
    if (persistMemory(value)) resetPersonForm();
  } catch (error) {
    alert(error.message || 'Could not import that file.');
  } finally {
    event.target.value = '';
  }
});
$('#clear-memory').addEventListener('click', () => {
  if (!memory.people.length && !memory.episodes.length) return;
  if (
    !confirm(
      'Permanently clear all saved people and episodes from this browser? Export a backup first if you may want them later.',
    )
  )
    return;
  if (persistMemory({ people: [], episodes: [] })) {
    resetPersonForm();
    refreshPersonSelect('');
  }
});

renderMemory();
$('#memory-status').textContent = memoryStore.warning;
setInterval(() => fetch('/__ping', { cache: 'no-store' }).catch(() => {}), 30_000);
